from ytbot.ui import ui
from ytbot.llm import (
    get_settings,
    Settings
)

if __name__ == "__main__":
    settings: Settings = get_settings()
    ui.launch_ui(settings)
