from autocar.common.types import DriveCommand


def encode_command(command: DriveCommand) -> str:
    if command.brake:
        return "STOP\n"
    return f"DRIVE {int(command.speed)} {int(command.steering)}\n"
