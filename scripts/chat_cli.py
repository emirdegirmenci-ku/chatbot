import argparse
import json
import sys

import requests


def ensure_session(base_url, session_id):
    if session_id:
        return session_id
    response = requests.post(f"{base_url}/api/session/create", timeout=30)
    response.raise_for_status()
    data = response.json()
    return data["session_id"]


def print_sources(sources):
    if not sources:
        return
    print("")
    print("Kaynaklar:")
    for item in sources:
        parts = []
        source = item.get("source") or ""
        if source:
            parts.append(source)
        page = item.get("page")
        if page is not None:
            parts.append(f"sayfa {page}")
        score = item.get("score")
        if isinstance(score, (int, float)):
            parts.append(f"skor {score:.3f}")
        if parts:
            print(" - " + ", ".join(parts))


def chat_once(base_url, session_id, message, stream):
    payload = {"session_id": session_id, "message": message, "stream": stream}
    if stream:
        response = requests.post(f"{base_url}/api/chat/stream", json=payload, stream=True, timeout=120)
        response.raise_for_status()
        summary = None
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line.decode("utf-8"))
            kind = data.get("type")
            if kind == "token":
                text = data.get("data", "")
                print(text, end="", flush=True)
            elif kind == "summary":
                summary = data
                print("")
        if summary:
            print_sources(summary.get("sources", []))
            return summary.get("session_id", session_id)
        return session_id
    response = requests.post(f"{base_url}/api/chat", json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    print(data.get("response", ""))
    print_sources(data.get("sources", []))
    return data.get("session_id", session_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--session-id")
    parser.add_argument("--message", required=True)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    try:
        session_id = ensure_session(args.base_url, args.session_id)
        final_session = chat_once(args.base_url, session_id, args.message, args.stream)
        print("")
        print("Kullanılan oturum:", final_session)
    except requests.exceptions.RequestException as exc:
        print(f"HTTP hatası: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
