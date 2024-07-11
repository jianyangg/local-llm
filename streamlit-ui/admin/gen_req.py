"""
This is used to automate the generation of a requirements file.
First, copy all imports into GPT and ask it to generate a python list of packages.
Then, copy this packages variable into the script and run it.
A requirements file will be generated. Do a second check to ensure all version numbers are in place.
Note: Self-defined packages may be mistakenly added, so do remember to remove them.
"""


import subprocess

# List of package names derived from the imports
# Use a set of all packages to avoid duplicates.
packages = [
    "flask",
    "pprint",  # part of the Python standard library
    "workflow",  # custom or a specific package, needs clarification
    "traceback",  # part of the Python standard library
    "langchain_core",
    "langchain_community",
    "app_config",  # custom or a specific package, needs clarification
    "termcolor",
    "GraphState",  # custom or a specific package, needs clarification
    "CustomLLM",  # custom or a specific package, needs clarification
    "flashrank",
    "typing",  # part of the Python standard library
    "typing_extensions",
    "langgraph"
]



# Function to get the version of a package
def get_package_version(package):
    try:
        output = subprocess.check_output(f"conda list {package}", shell=True).decode("utf-8")
        # print(output)
        for line in output.split("\n"):
            if line.startswith(package):
                return line.split()[1]
    except subprocess.CalledProcessError:
        return None

# Generate requirements.txt content
requirements = []
for package in packages:
    version = get_package_version(package)
    if version:
        requirements.append(f"{package}=={version}")
    else:
        requirements.append(package)

# Write to requirements.txt
with open("requirements.txt", "w") as file:
    file.write("\n".join(requirements))

print("requirements.txt generated successfully.")
