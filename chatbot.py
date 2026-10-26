import base64
import hashlib
import os
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import groq
import streamlit as st


APP_TITLE = "Chat Space"
LOCAL_WORKSPACE = "local"
CHAT_MODEL = os.environ.get(
    "GROQ_CHAT_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
)
TRANSCRIPTION_MODEL = os.environ.get(
    "GROQ_TRANSCRIPTION_MODEL", "whisper-large-v3-turbo"
)
ALLOWED_IMAGE_TYPES = ("jpg", "jpeg", "png", "webp")
AUDIO_EXTENSIONS = {
    "audio/flac": "flac",
    "audio/m4a": "m4a",
    "audio/mp3": "mp3",
    "audio/mp4": "mp4",
    "audio/mpeg": "mp3",
    "audio/ogg": "ogg",
    "audio/opus": "opus",
    "audio/wav": "wav",
    "audio/wave": "wav",
    "audio/webm": "webm",
    "audio/x-flac": "flac",
    "audio/x-m4a": "m4a",
    "audio/x-wav": "wav",
}
DATABASE_PATH = Path(
    os.environ.get(
        "CHATBOT_DB_PATH", str(Path(__file__).with_name("chatbot_history.sqlite3"))
    )
).expanduser()


@contextmanager
def database() -> Generator[sqlite3.Connection, None, None]:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database() -> None:
    with database() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_salt BLOB NOT NULL,
                password_hash BLOB NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                image BLOB,
                image_mime TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS chats_by_user
                ON chats(username, updated_at DESC);
            CREATE INDEX IF NOT EXISTS messages_by_chat
                ON messages(chat_id, id);
            """
        )
        connection.execute(
            "INSERT OR IGNORE INTO users "
            "(username, password_salt, password_hash) VALUES (?, ?, ?)",
            (LOCAL_WORKSPACE, b"", b""),
        )
        connection.execute(
            "UPDATE chats SET username = ? WHERE username != ?",
            (LOCAL_WORKSPACE, LOCAL_WORKSPACE),
        )


def create_chat(username: str) -> str:
    chat_id = uuid.uuid4().hex
    with database() as connection:
        connection.execute(
            "INSERT INTO chats (id, username, title) VALUES (?, ?, ?)",
            (chat_id, username, "New chat"),
        )
    return chat_id


def list_chats(username: str) -> list[sqlite3.Row]:
    with database() as connection:
        return connection.execute(
            "SELECT id, title FROM chats WHERE username = ? "
            "ORDER BY updated_at DESC, created_at DESC",
            (username,),
        ).fetchall()


def load_messages(chat_id: str, username: str) -> list[sqlite3.Row]:
    with database() as connection:
        return connection.execute(
            "SELECT messages.role, messages.content, messages.image, "
            "messages.image_mime FROM messages "
            "JOIN chats ON chats.id = messages.chat_id "
            "WHERE messages.chat_id = ? AND chats.username = ? "
            "ORDER BY messages.id",
            (chat_id, username),
        ).fetchall()


def save_exchange(
    chat_id: str,
    username: str,
    prompt: str,
    answer: str,
    image: bytes | None,
    image_mime: str | None,
) -> None:
    with database() as connection:
        chat = connection.execute(
            "SELECT title FROM chats WHERE id = ? AND username = ?",
            (chat_id, username),
        ).fetchone()
        if chat is None:
            raise ValueError("The selected chat does not belong to this account.")
        if chat["title"] == "New chat":
            title = prompt.strip() or "Photo conversation"
            title = title[:48] + ("…" if len(title) > 48 else "")
            connection.execute(
                "UPDATE chats SET title = ? WHERE id = ?", (title, chat_id)
            )
        connection.executemany(
            "INSERT INTO messages (chat_id, role, content, image, image_mime) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (chat_id, "user", prompt, image, image_mime),
                (chat_id, "assistant", answer, None, None),
            ],
        )
        connection.execute(
            "UPDATE chats SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (chat_id,),
        )


def image_data_url(image: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def request_answer(
    client: groq.Groq,
    history: list[sqlite3.Row],
    prompt: str,
    image: bytes | None,
    image_mime: str | None,
) -> str:
    messages: list[dict[str, object]] = [
        {
            "role": "system",
            "content": "You are a helpful, clear, and friendly assistant.",
        }
    ]
    for item in history[-20:]:
        content: str | list[dict[str, object]] = item["content"]
        if item["image"] is not None and item["image_mime"]:
            content = [
                {"type": "text", "text": item["content"]},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url(item["image"], item["image_mime"])
                    },
                },
            ]
        messages.append({"role": item["role"], "content": content})

    current_content: str | list[dict[str, object]] = prompt
    if image is not None and image_mime:
        current_content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": image_data_url(image, image_mime)},
            },
        ]
    messages.append({"role": "user", "content": current_content})

    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=2048,
    )
    answer = completion.choices[0].message.content
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("The AI returned an empty response.")
    return answer


def transcribe_audio(
    client: groq.Groq, audio_bytes: bytes, mime_type: str
) -> str:
    normalized_mime = mime_type.split(";", maxsplit=1)[0].strip().lower()
    extension = AUDIO_EXTENSIONS.get(normalized_mime)
    if extension is None:
        raise ValueError(f"Unsupported recording format: {mime_type or 'unknown'}.")
    transcript = client.audio.transcriptions.create(
        file=(f"voice-message.{extension}", audio_bytes, normalized_mime),
        model=TRANSCRIPTION_MODEL,
        response_format="text",
    )
    return str(transcript).strip()


def render_message(message: sqlite3.Row) -> None:
    with st.chat_message(message["role"]):
        if message["content"]:
            st.markdown(message["content"])
        if message["image"] is not None:
            st.image(message["image"], caption="Attached image", width=420)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="💬", layout="centered")
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
            --background-color: #0e1117;
            --secondary-background-color: #171b24;
            --text-color: #f1f3f8;
        }
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            background: #0e1117;
            color: #f1f3f8;
        }
        [data-testid="stHeader"] { background: rgba(14, 17, 23, .9); }
        [data-testid="stSidebar"] { background: #171b24; }
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stMarkdown"],
        [data-testid="stCaption"] { color: #d5d9e2; }
        [data-testid="stChatMessage"],
        [data-testid="stForm"],
        [data-testid="stFileUploader"],
        [data-testid="stAudioInput"],
        [data-testid="stCameraInput"] {
            background: #171b24;
            border-color: #303746;
        }
        [data-testid="stChatInput"] textarea,
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-testid="stSelectbox"] [data-baseweb="select"] > div {
            background: #171b24;
            color: #f1f3f8;
            border-color: #394150;
        }
        [data-testid="stTabs"] button { color: #d5d9e2; }
        [data-testid="stButton"] button,
        [data-testid="stFormSubmitButton"] button {
            background: #202633;
            color: #f1f3f8;
            border-color: #394150;
        }
        [data-testid="stButton"] button:hover,
        [data-testid="stFormSubmitButton"] button:hover {
            background: #2a3343;
            border-color: #59677d;
            color: #fff;
        }
        .block-container { max-width: 900px; padding-top: 2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    initialize_database()

    username = LOCAL_WORKSPACE
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    with st.sidebar:
        st.title("💬 Chat Space")
        st.caption("Your local chat workspace")
        if st.button("＋ New chat", use_container_width=True):
            st.session_state["selected_chat_id"] = create_chat(username)
            st.rerun()
        chats = list_chats(username)
        chat_ids = [chat["id"] for chat in chats]
        chat_titles = {chat["id"]: chat["title"] for chat in chats}
        selected_id = st.session_state.get("selected_chat_id")
        if selected_id not in chat_ids and chat_ids:
            selected_id = chat_ids[0]
            st.session_state["selected_chat_id"] = selected_id
        if chat_ids:
            selected_id = st.selectbox(
                "Your conversations",
                chat_ids,
                index=chat_ids.index(selected_id),
                format_func=lambda chat_id: chat_titles[chat_id],
                label_visibility="collapsed",
            )
            st.session_state["selected_chat_id"] = selected_id
        if api_key:
            st.caption("AI is ready.")
        else:
            st.warning("Set the GROQ_API_KEY environment variable to enable AI.")

    if not chat_ids:
        selected_id = create_chat(username)
        st.session_state["selected_chat_id"] = selected_id
        st.rerun()

    st.title("What can I help with?")
    st.caption("Ask a question, record a voice message, or share a photo.")
    if not api_key:
        st.info(
            "To enable replies and voice transcription, set **GROQ_API_KEY** "
            "in your environment and restart Streamlit."
        )

    history = load_messages(selected_id, username)
    for message in history:
        render_message(message)

    client = groq.Groq(api_key=api_key) if api_key else None
    audio = st.audio_input(
        "Record a voice message", key=f"voice_recording_{username}"
    )
    if audio is not None:
        audio_bytes = audio.getvalue()
        digest = hashlib.sha256(audio_bytes).hexdigest()
        if digest != st.session_state.get("audio_digest"):
            if client is None:
                st.session_state["audio_digest"] = digest
                st.warning("Set GROQ_API_KEY before recording a voice message.")
            else:
                with st.spinner("Transcribing your recording..."):
                    try:
                        transcript = transcribe_audio(
                            client, audio_bytes, audio.type or "audio/wav"
                        )
                    except (groq.APIError, ValueError) as error:
                        st.error(f"Voice transcription failed: {error}")
                    else:
                        st.session_state["audio_digest"] = digest
                        if transcript:
                            st.session_state["draft"] = transcript
                            st.success("Transcript added below. Review it before sending.")
                        else:
                            st.warning("No speech was detected in that recording.")

    uploaded_image = st.file_uploader(
        "Upload a photo",
        type=ALLOWED_IMAGE_TYPES,
        accept_multiple_files=False,
        key=f"photo_upload_{username}",
    )
    camera_image = st.camera_input(
        "Or take a photo", key=f"camera_photo_{username}"
    )
    image_source = camera_image if camera_image is not None else uploaded_image
    image_bytes = image_source.getvalue() if image_source is not None else None
    image_mime = image_source.type if image_source is not None else None
    if image_bytes is not None:
        st.image(image_bytes, caption="Image ready to send", width=240)

    if st.session_state.pop("clear_draft_on_next_run", False):
        st.session_state["draft"] = ""
    prompt = st.text_area(
        "Message",
        key="draft",
        height=100,
        placeholder="Message Chat Space...",
    )
    if st.button("Send", type="primary", use_container_width=True):
        prompt = prompt.strip()
        if not prompt and image_bytes is not None:
            prompt = "Please describe this image."
        if not prompt and image_bytes is None:
            st.warning("Write a message or attach a photo first.")
        elif client is None:
            st.error("Set GROQ_API_KEY before sending a message.")
        else:
            with st.spinner("Thinking..."):
                try:
                    answer = request_answer(
                        client, history, prompt, image_bytes, image_mime
                    )
                except (groq.APIError, ValueError) as error:
                    st.error(f"The AI request failed: {error}")
                else:
                    save_exchange(
                        selected_id,
                        username,
                        prompt,
                        answer,
                        image_bytes,
                        image_mime,
                    )
                    st.session_state["clear_draft_on_next_run"] = True
                    st.rerun()

    st.caption(
        "Chats and attached photos are saved locally. Photos are sent to Groq "
        "with your prompt; voice recordings are sent to Groq for transcription."
    )


if __name__ == "__main__":
    main()
