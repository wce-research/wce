from app.session import Session
from language.command import CommandParseError


def main():
    session = Session()

    print("WCE — World Configuration Engine")
    print("Tape 'quit' pour quitter.")
    print()

    while True:
        try:
            text = input("> ").strip()

        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not text:
            continue

        if text.lower() in {
            "quit",
            "exit",
            "quitter",
        }:
            break

        try:
            response = session.execute(text)
            print(response)

        except Exception as error:
            print(
                f"ERREUR: {error}"
            )


if __name__ == "__main__":
    main()
