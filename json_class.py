import json


class IPCameraDatabase:
    def __init__(self, filename):
        self.filename = filename
        self.data = self.load_data()

    def load_data(self):
        try:
            with open(self.filename, "r") as file:
                data = json.load(file)
            print("Loaded")
            return data
        except FileNotFoundError:
            return {}

    def save_data(self):
        with open(self.filename, "w") as file:
            json.dump(self.data, file, indent=4)

    def add_ip_camera(self, ip_camera, lines, overwrite=False):
        if ip_camera in self.data and not overwrite:
            print("IP camera already exists. Not overwriting data.")
            return False
        self.data[ip_camera] = lines
        self.save_data()
        print("IP camera added successfully.")
        return True

    def edit_ip_camera(self, ip_camera, new_lines):
        if ip_camera in self.data:
            self.data[ip_camera] = new_lines
            self.save_data()
            return True
        return False

    def delete_ip_camera(self, ip_camera):
        if ip_camera in self.data:
            del self.data[ip_camera]
            self.save_data()
            return True
        return False

    def get_data_by_ip_camera(self, ip_camera):
        return self.data.get(ip_camera, None)

    def display_data(self):
        print("Updated data:")
        print(self.data)


if __name__ == "__main__":

    # Example usage:
    filename = "list_line.json"

    # Create instance of IPCameraDatabase
    ip_camera_db = IPCameraDatabase(filename)
    ip_camera_db.display_data()

    # Add new ip_camera and lines without overwriting
    ip_camera_db.add_ip_camera("3.3.3.3", [[0, 360], [1920, 360]])
    ip_camera_db.display_data()

    # Try to add new ip_camera and lines without overwriting (should fail)
    ip_camera_db.add_ip_camera("3.3.3.3", [[0, 200], [1920, 200]])
    ip_camera_db.display_data()

    # Add new ip_camera and lines with overwriting
    ip_camera_db.add_ip_camera("3.3.3.3", [[0, 200], [1920, 200]], overwrite=True)
    ip_camera_db.display_data()

    # Edit existing ip_camera to new lines
    if ip_camera_db.edit_ip_camera("1.1.1.1", [[0, 200], [1920, 200]]):
        print("Edit successful.")
    else:
        print("Edit failed. IP camera not found.")
    ip_camera_db.display_data()

    # Delete an IP camera
    if ip_camera_db.delete_ip_camera("3.3.3.3"):
        print("Deletion successful.")
    else:
        print("Deletion failed. IP camera not found.")
    ip_camera_db.display_data()

    # Get data for a specific IP camera
    print(ip_camera_db.get_data_by_ip_camera("1.1.1.1"))
