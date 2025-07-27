import json
import sys
from Environment.factory_data_classes import SchedulingInstanceData


def parse_inputdata_to_factory(json_path: str):
    with open(json_path, 'r') as f:
        data = json.load(f)
    # The root key is 'scheduling_instance'
    scheduling_instance_dict = data["scheduling_instance"]
    scheduling_instance = SchedulingInstanceData.from_dict(scheduling_instance_dict)
    return scheduling_instance


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_inputdata_to_factory.py <inputdata.json>")
        sys.exit(1)
    json_path = sys.argv[1]
    scheduling_instance = parse_inputdata_to_factory(json_path)
    print("Parsed SchedulingInstanceData:")
    print(scheduling_instance) 