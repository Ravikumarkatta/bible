
import json
from jsonschema import validate
from jsonschema.exceptions import ValidationError

def validate_config(config_file, schema_file):
    """
    Validates a JSON configuration file against a JSON schema.
    """
    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        with open(schema_file, 'r') as f:
            schema = json.load(f)

        validate(instance=config_data, schema=schema)
        print(f"Configuration file '{config_file}' is valid.")
        return True
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {config_file} or {schema_file}: {e}")
        return False
    except ValidationError as e:
        print(f"Error: Configuration validation failed: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

if __name__ == '__main__':
    # Example usage:
    config_file = 'config/data_config.json'
    schema_file = 'config/data_config_schema.json'  # Create this file
    validate_config(config_file, schema_file)