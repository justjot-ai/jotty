"""
Voice routes - TTS, STT, voice chat, streaming variants.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def register_voice_routes(app, api):
    from fastapi import File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
    from pydantic import BaseModel

    @app.get("/api/voice/voices")
    async def list_voices():
        """Get available TTS voices."""
        from .voice import get_voice_processor

        processor = get_voice_processor()
        return {"voices": processor.get_available_voices()}

    @app.get("/api/voice/config")
    async def get_voice_config():
        """
        Get voice processing configuration.

        Returns current settings for:
        - Local Whisper (LOCAL_WHISPER env var)
        - Speculative TTS (SPECULATIVE_TTS env var)
        - WebSocket voice (WEBSOCKET_VOICE env var)
        - Whisper model size
        """
        import os

        use_local_whisper = os.environ.get("LOCAL_WHISPER", "0") == "1"
        return {
            "local_whisper": use_local_whisper,
            "whisper_model": os.environ.get("WHISPER_MODEL", "base") if use_local_whisper else None,
            "speculative_tts": os.environ.get("SPECULATIVE_TTS", "0") == "1",
            "websocket_voice": os.environ.get("WEBSOCKET_VOICE", "0") == "1",
            "websocket_url": "/ws/voice/{session_id}",
        }

    @app.post("/api/voice/config")
    async def set_voice_config(request: dict):
        """
        Update voice processing configuration at runtime.

        Request: {"local_whisper": true, "speculative_tts": true, "whisper_model": "small"}
        """
        import os

        if "local_whisper" in request:
            os.environ["LOCAL_WHISPER"] = "1" if request["local_whisper"] else "0"
        if "speculative_tts" in request:
            os.environ["SPECULATIVE_TTS"] = "1" if request["speculative_tts"] else "0"
        if "websocket_voice" in request:
            os.environ["WEBSOCKET_VOICE"] = "1" if request["websocket_voice"] else "0"
        if "whisper_model" in request:
            os.environ["WHISPER_MODEL"] = request["whisper_model"]

        # Return updated config
        return await get_voice_config()

    @app.post("/api/voice/tts")
    async def text_to_speech_endpoint(request: dict):
        """
        Convert text to speech.

        Request: {"text": "Hello", "voice": "en-US-AvaNeural"}
        Returns: Audio file (MP3)
        """
        from fastapi.responses import Response

        from .voice import get_voice_processor

        text = request.get("text", "")
        voice = request.get("voice")

        if not text:
            raise HTTPException(status_code=400, detail="Text is required")

        processor = get_voice_processor()
        audio_data = await processor.text_to_speech(text, voice)

        if not audio_data:
            raise HTTPException(status_code=500, detail="TTS failed")

        return Response(
            content=audio_data,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=speech.mp3"},
        )

    @app.post("/api/voice/stt")
    async def speech_to_text_endpoint(audio: UploadFile = File(...)):
        """
        Convert speech to text.

        Accepts audio file (webm, wav, mp3, etc.)
        Returns: {"text": "transcribed text"}
        """
        from .voice import get_voice_processor

        content = await audio.read()
        mime_type = audio.content_type or "audio/webm"

        processor = get_voice_processor()
        text, confidence = await processor.speech_to_text(content, mime_type)

        return {"text": text, "confidence": confidence, "success": bool(text)}

    @app.post("/api/voice/chat/audio")
    async def voice_chat_audio_endpoint(
        audio: UploadFile = File(...),
        session_id: Optional[str] = Form(None),
        voice: Optional[str] = Form(None),
    ):
        """
        Full voice-to-voice chat - returns raw audio.

        For direct audio playback in browser. Returns MP3 with metadata in headers.
        For JSON response with base64 audio, use /api/voice/chat instead.
        """
        from fastapi.responses import Response

        from .voice import get_voice_processor

        session_id = session_id or str(uuid.uuid4())[:8]

        # Read audio
        audio_content = await audio.read()
        mime_type = audio.content_type or "audio/webm"

        processor = get_voice_processor()

        # 1. Speech to text
        user_text, _ = await processor.speech_to_text(audio_content, mime_type)
        if not user_text:
            # Return error audio
            error_audio = await processor.text_to_speech(
                "I couldn't understand that. Please try again.", voice
            )
            return Response(
                content=error_audio,
                media_type="audio/mpeg",
                headers={
                    "X-User-Text": "",
                    "X-Response-Text": "Could not transcribe",
                    "Content-Disposition": "inline; filename=response.mp3",
                },
            )

        # 2. Process with LLM
        result = await api.process_message(message=user_text, session_id=session_id)

        response_text = result.get("content", "I encountered an error processing your request.")

        # 3. Text to speech
        response_audio = await processor.text_to_speech(response_text, voice)

        if not response_audio:
            raise HTTPException(status_code=500, detail="TTS failed")

        # Return audio with metadata in headers
        import urllib.parse

        return Response(
            content=response_audio,
            media_type="audio/mpeg",
            headers={
                "X-User-Text": urllib.parse.quote(user_text[:500]),
                "X-Response-Text": urllib.parse.quote(response_text[:500]),
                "X-Session-Id": session_id,
                "Content-Disposition": "inline; filename=response.mp3",
            },
        )

    @app.websocket("/ws/voice/{session_id}")
    async def websocket_voice_chat(websocket: WebSocket, session_id: str):
        """
        WebSocket endpoint for low-latency streaming voice chat.

        Features:
        - Real-time audio streaming (lower latency than HTTP)
        - Local Whisper support (if LOCAL_WHISPER=1)
        - Speculative TTS (if SPECULATIVE_TTS=1)
        - Confidence scores for transcription

        Client sends:
        - Binary audio chunks (accumulates until end_audio)
        - {"type": "config", "voice": "...", "speed": 1.0, "speculative": true}
        - {"type": "end_audio"} - triggers processing
        - {"type": "cancel"} - cancel current processing

        Server sends:
        - {"type": "config_ack", "voice": "...", "local_whisper": bool, "speculative_tts": bool}
        - {"type": "transcription", "text": "...", "confidence": 0.95}
        - {"type": "response_text", "text": "..."}
        - {"type": "audio_start"}
        - Binary audio chunks
        - {"type": "audio_end"}
        - {"type": "error", "message": "..."}
        """
        import os

        from .voice import VoiceConfig, get_voice_processor

        await websocket.accept()

        # Get config from environment
        use_local_whisper = os.environ.get("LOCAL_WHISPER", "0") == "1"
        use_speculative_tts = os.environ.get("SPECULATIVE_TTS", "0") == "1"

        processor = get_voice_processor()
        voice = None
        speed = 1.0
        audio_buffer = bytearray()
        cancelled = False

        try:
            # Send initial capabilities
            await websocket.send_json(
                {
                    "type": "capabilities",
                    "local_whisper": use_local_whisper,
                    "speculative_tts": use_speculative_tts,
                    "whisper_model": (
                        os.environ.get("WHISPER_MODEL", "base") if use_local_whisper else None
                    ),
                }
            )

            while True:
                message = await websocket.receive()

                if "text" in message:
                    import json

                    data = json.loads(message["text"])

                    if data.get("type") == "config":
                        voice = data.get("voice")
                        speed = data.get("speed", 1.0)
                        # Allow client to override speculative TTS
                        if "speculative" in data:
                            use_speculative_tts = data["speculative"]
                        await websocket.send_json(
                            {
                                "type": "config_ack",
                                "voice": voice,
                                "speed": speed,
                                "local_whisper": use_local_whisper,
                                "speculative_tts": use_speculative_tts,
                            }
                        )

                    elif data.get("type") == "cancel":
                        cancelled = True
                        audio_buffer.clear()
                        await websocket.send_json({"type": "cancelled"})

                    elif data.get("type") == "end_audio":
                        cancelled = False
                        if audio_buffer:
                            try:
                                # STT - with confidence
                                user_text, confidence = await processor.speech_to_text(
                                    bytes(audio_buffer), "audio/webm"
                                )
                                await websocket.send_json(
                                    {
                                        "type": "transcription",
                                        "text": user_text,
                                        "confidence": confidence,
                                    }
                                )

                                if user_text and not cancelled:
                                    # LLM
                                    result = await api.process_message(
                                        message=user_text, session_id=session_id
                                    )
                                    response_text = result.get("content", "")

                                    if cancelled:
                                        continue

                                    await websocket.send_json(
                                        {"type": "response_text", "text": response_text}
                                    )

                                    # TTS - stream chunks
                                    await websocket.send_json({"type": "audio_start"})

                                    if use_speculative_tts and hasattr(
                                        processor, "speculative_tts_stream"
                                    ):
                                        # Use speculative TTS for even lower latency
                                        # Split into sentences for parallel generation
                                        import re

                                        sentences = re.split(r"(?<=[.!?])\s+", response_text)
                                        for sentence in sentences:
                                            if cancelled:
                                                break
                                            if sentence.strip():
                                                audio = await processor.text_to_speech(
                                                    sentence, voice, speed
                                                )
                                                if audio:
                                                    await websocket.send_bytes(audio)
                                    else:
                                        # Standard streaming TTS
                                        async for chunk in processor.text_to_speech_stream(
                                            response_text, voice
                                        ):
                                            if cancelled:
                                                break
                                            await websocket.send_bytes(chunk)

                                    await websocket.send_json({"type": "audio_end"})

                            except Exception as e:
                                logger.error(f"Voice processing error: {e}")
                                await websocket.send_json({"type": "error", "message": str(e)})

                            audio_buffer.clear()

                elif "bytes" in message:
                    # Audio chunk - accumulate
                    audio_buffer.extend(message["bytes"])

        except Exception as e:
            logger.error(f"Voice WebSocket error: {e}")
        finally:
            audio_buffer.clear()

    # WebSocket endpoint
    @app.post("/api/voice/stt")
    async def speech_to_text(audio: UploadFile):
        """
        Convert speech audio to text using Groq Whisper (primary) or Deepgram (fallback).

        Accepts audio files (webm, wav, mp3, ogg, flac, m4a).
        Returns transcribed text.
        """
        from .voice import get_voice_processor

        processor = get_voice_processor()
        audio_data = await audio.read()
        mime_type = audio.content_type or "audio/webm"

        transcript, confidence = await processor.speech_to_text(audio_data, mime_type)

        return {
            "success": bool(transcript),
            "transcript": transcript,
            "confidence": confidence,
            "mime_type": mime_type,
        }

    @app.post("/api/voice/tts")
    async def text_to_speech(text: str = Form(...), voice: Optional[str] = Form(None)):
        """
        Convert text to speech using edge-tts (Microsoft neural voices).

        Args:
            text: Text to convert to speech (form field)
            voice: Optional voice ID (default: en-US-AvaNeural)

        Returns audio/mpeg stream.
        """
        from fastapi.responses import Response

        from .voice import get_voice_processor

        processor = get_voice_processor()
        audio_bytes = await processor.text_to_speech(text, voice)

        if not audio_bytes:
            raise HTTPException(status_code=500, detail="TTS generation failed")

        return Response(content=audio_bytes, media_type="audio/mpeg")

    @app.get("/api/voice/voices")
    async def list_voices():
        """List available TTS voices."""
        from .voice import VoiceProcessor

        return {"voices": VoiceProcessor.get_available_voices(), "default": "en-US-AvaNeural"}

    @app.post("/api/voice/chat")
    async def voice_chat(audio: UploadFile, session_id: Optional[str] = None):
        """
        Full voice-to-voice pipeline: STT -> LLM -> TTS.

        Processes audio input through:
        1. Speech-to-Text (Groq Whisper)
        2. LLM processing (via chat endpoint logic)
        3. Text-to-Speech (edge-tts)

        Returns JSON with text and base64-encoded audio.
        """
        import base64

        from .voice import get_voice_processor

        processor = get_voice_processor()
        audio_data = await audio.read()
        mime_type = audio.content_type or "audio/webm"

        # Define LLM processing function
        async def process_with_llm(user_text: str) -> str:
            result = await api.process_message(
                message=user_text, session_id=session_id or str(uuid.uuid4())[:8]
            )
            return result.get("content", "") if isinstance(result, dict) else str(result)

        user_text, response_text, response_audio = await processor.process_voice_message(
            audio_data, mime_type, process_with_llm
        )

        return {
            "success": True,
            "user_text": user_text,
            "response_text": response_text,
            "response_audio_base64": (
                base64.b64encode(response_audio).decode() if response_audio else None
            ),
            "audio_format": "audio/mpeg",
        }

    @app.post("/api/voice/chat/fast")
    async def voice_chat_fast(
        audio: UploadFile, session_id: Optional[str] = None, max_chars: int = 200
    ):
        """
        Optimized voice pipeline for minimum latency.

        Optimizations:
        - Truncates response at sentence boundary (max 200 chars default)
        - Uses 15% faster speech rate
        - Reduces overall latency by ~40%

        Returns JSON with text and base64-encoded audio.
        """
        import base64

        from .voice import get_voice_processor

        processor = get_voice_processor()
        audio_data = await audio.read()
        mime_type = audio.content_type or "audio/webm"

        async def process_with_llm(user_text: str) -> str:
            result = await api.process_message(
                message=user_text, session_id=session_id or str(uuid.uuid4())[:8]
            )
            return result.get("content", "") if isinstance(result, dict) else str(result)

        user_text, response_text, response_audio = await processor.process_voice_fast(
            audio_data, mime_type, process_with_llm, max_response_chars=max_chars
        )

        return {
            "success": True,
            "user_text": user_text,
            "response_text": response_text,
            "response_audio_base64": (
                base64.b64encode(response_audio).decode() if response_audio else None
            ),
            "audio_format": "audio/mpeg",
            "mode": "fast",
        }

    @app.post("/api/voice/chat/turbo")
    async def voice_chat_turbo(audio: UploadFile, session_id: Optional[str] = None):
        """
        Ultra-fast voice pipeline using Groq LLM (~2s total latency).

        Uses Groq for both STT (Whisper) and LLM (llama-3.1-8b-instant).
        Optimized for conversational voice chat where latency is critical.

        Latency breakdown:
        - STT (Groq Whisper): ~250ms
        - LLM (Groq llama-3.1-8b): ~180ms
        - TTS (edge-tts): ~700ms
        - Total: ~1.1-2.3s

        Returns raw audio/mpeg with X-User-Text and X-Response-Text headers.
        """
        import os
        from urllib.parse import quote

        import httpx
        from fastapi.responses import Response

        from .voice import get_voice_processor

        processor = get_voice_processor()
        audio_data = await audio.read()
        mime_type = audio.content_type or "audio/webm"

        # Ultra-fast LLM using Groq's llama-3.1-8b-instant
        async def process_with_groq_llm(user_text: str) -> str:
            groq_key = os.environ.get("GROQ_API_KEY")
            if not groq_key:
                return "I'm sorry, the fast response service is unavailable."

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {groq_key}",
                    },
                    json={
                        "model": "llama-3.1-8b-instant",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a helpful voice assistant. Keep responses brief and conversational (1-3 sentences).",
                            },
                            {"role": "user", "content": user_text},
                        ],
                        "max_tokens": 150,
                        "temperature": 0.7,
                    },
                )
                data = response.json()
                return (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "I couldn't process that.")
                )

        user_text, response_text, response_audio = await processor.process_voice_fast(
            audio_data, mime_type, process_with_groq_llm, max_response_chars=200
        )

        # Return raw audio with text in headers (for UI compatibility)
        return Response(
            content=response_audio or b"",
            media_type="audio/mpeg",
            headers={
                "X-User-Text": quote(user_text or ""),
                "X-Response-Text": quote(response_text or ""),
                "X-Mode": "turbo",
                "X-LLM": "groq/llama-3.1-8b-instant",
            },
        )

    @app.post("/api/voice/chat/stream")
    async def voice_chat_streaming(audio: UploadFile, session_id: Optional[str] = None):
        """
        Streaming voice pipeline for lower perceived latency.

        Returns audio sentence-by-sentence as Server-Sent Events.
        First event includes user_text, then response chunks follow.
        """
        import base64
        import json
        import re

        from fastapi.responses import StreamingResponse

        from .voice import get_voice_processor

        processor = get_voice_processor()
        audio_data = await audio.read()
        mime_type = audio.content_type or "audio/webm"

        async def process_with_llm(user_text: str) -> str:
            result = await api.process_message(
                message=user_text, session_id=session_id or str(uuid.uuid4())[:8]
            )
            return result.get("content", "") if isinstance(result, dict) else str(result)

        async def generate_sse():
            # 1. Speech to text
            user_text, confidence = await processor.speech_to_text(audio_data, mime_type)

            if not user_text:
                error_audio = await processor.text_to_speech("I couldn't understand that.")
                data = {
                    "user_text": "",
                    "confidence": 0.0,
                    "text": "I couldn't understand that.",
                    "audio_base64": base64.b64encode(error_audio).decode() if error_audio else None,
                }
                yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"
                return

            # 2. Send first event with user_text immediately
            yield f"data: {json.dumps({'user_text': user_text, 'confidence': confidence, 'text': '', 'audio_base64': None})}\n\n"

            # 3. Process with LLM
            response_text = await process_with_llm(user_text)

            # 4. Split into sentences and stream TTS for each
            sentences = re.split(r"(?<=[.!?])\s+", response_text)

            for sentence in sentences:
                if sentence.strip():
                    audio_chunk = await processor.text_to_speech(sentence)
                    data = {
                        "text": sentence + " ",
                        "audio_base64": (
                            base64.b64encode(audio_chunk).decode() if audio_chunk else None
                        ),
                    }
                    yield f"data: {json.dumps(data)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(generate_sse(), media_type="text/event-stream")

    @app.post("/api/voice/chat/stream/turbo")
    async def voice_chat_streaming_turbo(
        audio: UploadFile, session_id: Optional[str] = None, voice: Optional[str] = None
    ):
        """
        Ultra-fast streaming voice pipeline using Groq LLM.

        Combines Groq Whisper STT + Groq LLM + parallel TTS.
        First sentence plays in ~1s, subsequent sentences ready immediately.

        Optimization: TTS for all sentences generated in parallel.
        """
        import asyncio
        import base64
        import json
        import os
        import re

        import httpx
        from fastapi.responses import StreamingResponse

        from .voice import get_voice_processor

        processor = get_voice_processor()
        audio_data = await audio.read()
        mime_type = audio.content_type or "audio/webm"
        tts_voice = voice or "en-US-AvaNeural"

        async def process_with_groq_llm(user_text: str) -> str:
            groq_key = os.environ.get("GROQ_API_KEY")
            if not groq_key:
                return "I'm sorry, the fast response service is unavailable."

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {groq_key}",
                    },
                    json={
                        "model": "llama-3.1-8b-instant",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a helpful voice assistant. Keep responses brief and conversational (2-4 sentences).",
                            },
                            {"role": "user", "content": user_text},
                        ],
                        "max_tokens": 200,
                        "temperature": 0.7,
                    },
                )
                data = response.json()
                return (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "I couldn't process that.")
                )

        async def generate_sse():
            # 1. Speech to text (Groq Whisper - ~250ms, or local Whisper)
            user_text, confidence = await processor.speech_to_text(audio_data, mime_type)

            if not user_text:
                error_audio = await processor.text_to_speech("I couldn't understand that.")
                data = {
                    "user_text": "",
                    "confidence": 0.0,
                    "text": "I couldn't understand that.",
                    "audio_base64": base64.b64encode(error_audio).decode() if error_audio else None,
                }
                yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"
                return

            # 2. Send user_text immediately with confidence
            yield f"data: {json.dumps({'user_text': user_text, 'confidence': confidence, 'text': '', 'audio_base64': None})}\n\n"

            # 3. Process with Groq LLM (~180ms)
            response_text = await process_with_groq_llm(user_text)

            # 4. Split into sentences
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", response_text) if s.strip()]

            if not sentences:
                yield "data: [DONE]\n\n"
                return

            # 5. PARALLEL TTS: Generate audio for ALL sentences concurrently
            # This reduces total TTS time from 700ms * N to ~700ms (parallel)
            async def generate_tts(sentence: str) -> tuple:
                audio = await processor.text_to_speech(sentence, tts_voice)
                return (sentence, audio)

            # Start all TTS tasks in parallel
            tts_tasks = [generate_tts(s) for s in sentences]

            # Stream results as they complete, but maintain order
            # Use asyncio.gather to run in parallel, preserving order
            results = await asyncio.gather(*tts_tasks)

            # Yield each result in order
            for sentence, audio_chunk in results:
                data = {
                    "text": sentence + " ",
                    "audio_base64": base64.b64encode(audio_chunk).decode() if audio_chunk else None,
                }
                yield f"data: {json.dumps(data)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(generate_sse(), media_type="text/event-stream")

    @app.post("/api/voice/chat/stream/ultra")
    async def voice_chat_streaming_ultra(
        audio: UploadFile,
        session_id: Optional[str] = None,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
    ):
        """
        Ultra-low-latency streaming: TTS starts BEFORE LLM finishes.

        Streams LLM tokens, generates TTS as soon as each sentence completes.
        First audio plays ~500ms after first sentence is generated.
        """
        import base64
        import json
        import os
        import re

        import httpx
        from fastapi.responses import StreamingResponse

        from .voice import get_voice_processor

        processor = get_voice_processor()
        audio_data = await audio.read()
        mime_type = audio.content_type or "audio/webm"
        tts_voice = voice or "en-US-AvaNeural"
        tts_speed = speed or 1.0

        async def generate_sse():
            # 1. Speech to text (now returns tuple with confidence)
            user_text, confidence = await processor.speech_to_text(audio_data, mime_type)

            if not user_text:
                error_audio = await processor.text_to_speech(
                    "I couldn't understand that.", tts_voice, tts_speed
                )
                yield f"data: {json.dumps({'user_text': '', 'text': 'Error', 'confidence': 0.0, 'audio_base64': base64.b64encode(error_audio).decode() if error_audio else None})}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Send user text immediately with confidence
            yield f"data: {json.dumps({'user_text': user_text, 'confidence': confidence, 'text': '', 'audio_base64': None})}\n\n"

            # 2. Stream from Groq LLM
            groq_key = os.environ.get("GROQ_API_KEY")
            if not groq_key:
                error_audio = await processor.text_to_speech(
                    "Fast response service unavailable.", tts_voice, tts_speed
                )
                yield f"data: {json.dumps({'text': 'Error', 'audio_base64': base64.b64encode(error_audio).decode() if error_audio else None})}\n\n"
                yield "data: [DONE]\n\n"
                return

            sentence_buffer = ""
            full_response = ""

            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream(
                    "POST",
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {groq_key}",
                    },
                    json={
                        "model": "llama-3.1-8b-instant",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a helpful voice assistant. Keep responses brief (2-3 sentences). IMPORTANT: Always respond in the SAME LANGUAGE the user speaks. If they speak Spanish, respond in Spanish. If French, respond in French. Match their language exactly.",
                            },
                            {"role": "user", "content": user_text},
                        ],
                        "max_tokens": 150,
                        "temperature": 0.7,
                        "stream": True,
                    },
                ) as response:
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue

                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")

                            if content:
                                sentence_buffer += content
                                full_response += content

                                # Check for sentence boundary
                                sentence_match = re.match(
                                    r"^(.*?[.!?])\s*(.*)$", sentence_buffer, re.DOTALL
                                )
                                if sentence_match:
                                    complete_sentence = sentence_match.group(1).strip()
                                    sentence_buffer = sentence_match.group(2)

                                    if complete_sentence:
                                        # Generate TTS immediately for this sentence
                                        audio_chunk = await processor.text_to_speech(
                                            complete_sentence, tts_voice, tts_speed
                                        )
                                        yield f"data: {json.dumps({'text': complete_sentence + ' ', 'audio_base64': base64.b64encode(audio_chunk).decode() if audio_chunk else None})}\n\n"

                        except json.JSONDecodeError:
                            continue

            # Send any remaining text
            if sentence_buffer.strip():
                audio_chunk = await processor.text_to_speech(
                    sentence_buffer.strip(), tts_voice, tts_speed
                )
                yield f"data: {json.dumps({'text': sentence_buffer.strip(), 'audio_base64': base64.b64encode(audio_chunk).decode() if audio_chunk else None})}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(generate_sse(), media_type="text/event-stream")

    # Explicit routes for static files (ensure they work)
    static_dir = Path(__file__).parent / "static"
