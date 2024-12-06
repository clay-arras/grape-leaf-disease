import os


def deleteFile(fileName):
    if os.path.exists(fileName):
        os.remove(fileName)


THRESHOLD = 0.9999

hasDeleted = set()
filenames = [
    "similarity/similarity__BlackRot.txt",
    "similarity/similarity__ESCA.txt",
    "similarity/similarity__Healthy.txt",
    "similarity/similarity__LeafBlight.txt"
]

for filename in filenames:
    with open(filename, "r") as file:
        while True:
            line = file.readline()

            if not line:
                break

            line = line.strip()
            parts = line.split(",")

            if len(parts) != 3:
                print(f"File error, too many commas `{line}`")
                continue

            if float(parts[0]) < THRESHOLD:
                break

            if parts[1] not in hasDeleted:
                hasDeleted.add(parts[2])
                deleteFile(parts[2])
