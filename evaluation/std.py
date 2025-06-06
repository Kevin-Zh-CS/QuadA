import re
import sys

from utils import validate_noise_source_name


# Modify model source code in
# ~/.cache/huggingface/modules/transformers_modules, or
# ./transformers/src/transformers/models/**/modeling_*.py
if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise ValueError(
            "Usage: python _increment_std.py <noise_source> <noise_std_increment> <noise_std_max> <model_file_path>"
        )

    NOISE_SOURCE = str(sys.argv[1])
    validate_noise_source_name(NOISE_SOURCE)

    NOISE_STD_INCREMENT = float(sys.argv[2])
    NOISE_STD_MAX = float(sys.argv[3])
    NOISE_MODEL_FILE_PATH = str(sys.argv[4])

    NOISE_STD_CONSTANT_NAME = f"{NOISE_SOURCE}_NOISE_STD"

    try:
        with open(NOISE_MODEL_FILE_PATH, "r") as file:
            lines = file.readlines()

        constant_pattern = re.compile(rf"^{NOISE_STD_CONSTANT_NAME}\s*=\s*(.+)")

        update_status = 0
        update_value = None

        for i, line in enumerate(lines):
            match = constant_pattern.match(line.strip())

            if match:
                update_status = 1

                old_value = match.group(1).strip()
                try:
                    old_value = float(old_value)
                except ValueError:
                    raise ValueError(
                        f"Constant '{NOISE_STD_CONSTANT_NAME}' is not a numeric value"
                    )
                update_value = round(old_value + NOISE_STD_INCREMENT, 4)

                if update_value > NOISE_STD_MAX:
                    update_status = -1
                    lines[i] = f"{NOISE_STD_CONSTANT_NAME} = {NOISE_STD_MAX}\n"
                else:
                    lines[i] = f"{NOISE_STD_CONSTANT_NAME} = {update_value}\n"

                break

        if update_status == 0:
            raise ValueError(
                f"Noise std constant '{NOISE_STD_CONSTANT_NAME}' not found in the file"
            )
        else:
            with open(NOISE_MODEL_FILE_PATH, "w") as file:
                file.writelines(lines)

            if update_status == 1:
                if NOISE_STD_MAX == 0.00:
                    print(
                        f"Resetting '{NOISE_STD_CONSTANT_NAME}' to the maximum value of {NOISE_STD_MAX}"
                    )
                else:
                    print(
                        f"Changed '{NOISE_STD_CONSTANT_NAME}' from {old_value} to {update_value}"
                    )
            elif NOISE_STD_MAX == 0.00:
                print(
                    f"Resetting '{NOISE_STD_CONSTANT_NAME}' to the maximum value of {NOISE_STD_MAX}"
                )
            else:
                raise ValueError(
                    f"'{NOISE_STD_CONSTANT_NAME}' reached the maximum noise std value of {NOISE_STD_MAX}"
                )

    except Exception as e:
        print(e)
