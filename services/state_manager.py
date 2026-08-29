"""Runtime state tracking for the inspection application."""

from enum import Enum


class ProgramState(Enum):
    """High-level states shown to the operator."""

    STARTING = "STARTING"
    WAITING_SOURCE = "WAITING_SOURCE"
    READY = "READY"
    DETECTING = "DETECTING"
    ERROR = "ERROR"


class StateManager:
    """Tracks the current system state and an optional error message."""

    def __init__(self):
        self.state = ProgramState.STARTING
        self.message = ""

    def set_state(self, state, message=""):
        self.state = state
        self.message = message

    def get_state(self):
        return self.state

    def get_message(self):
        return self.message

    def is_ready(self):
        return self.state == ProgramState.READY

    def is_error(self):
        return self.state == ProgramState.ERROR
