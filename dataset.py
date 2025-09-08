import os

# Path to the labels folder inside gascylinder-1
label_dirs = [r"C:\Users\srish\yolov5\yolov5\CylinDeRS-1\valid\labels"]

for label_dir in label_dirs:
    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(label_dir, filename)
            with open(file_path) as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts and parts[0] == "0":
                    parts[0] = "1"
                new_lines.append(" ".join(parts) + "\n")

            with open(file_path, "w") as f:
                f.writelines(new_lines)

print("✅ Labels updated: class 0 → class 1")
