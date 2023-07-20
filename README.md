# Big Data and BI Workshop

This README describes how to set up a Python virtual environment and install the necessary packages to run this project. 

## Prerequisites

Before you begin, ensure you have met the following requirements:

* You have installed Python 3.6 or later. You can download Python [here](https://www.python.org/downloads/).

* You have a basic familiarity with command line operations.

## Setting Up the Virtual Environment

To set up the virtual environment and install the necessary packages, follow these steps:

### 1. **Clone this repository to your local machine**:

Via command line:

```bash
git clone https://github.com/NielsSimmet/BigDataBI-Workshop.git
```

Or via GitHub Desktop:

https://github.com/NielsSimmet/BigDataBI-Workshop.git

### 2. **Navigate into the cloned repository**:

Via command line:

```bash
cd BigDataBI-Workshop
```

Or via GitHub Desktop:

"Open in Visual Studio Code"

### 3. **Create a virtual environment**:

Enter this into commandline after navigating to the repo or open a new terminal in VSCode for the repo and enter it there:

```bash
python3 -m venv env
```

This creates a new virtual environment in a folder named `env`.

### 4. **Activate the virtual environment**:

Enter this into commandline after navigating to the repo or open a new terminal in VSCode for the repo and enter it there:

- On macOS and Linux:

    ```bash
    source env/bin/activate
    ```

- On Windows:

    ```bash
    .\env\Scripts\activate
    ```

### 5. **Install the necessary packages**:

Enter this into commandline after navigating to the repo or open a new terminal in VSCode for the repo and enter it there:

```bash
pip install -r requirements.txt
```

You're now ready to run the project! Remember to activate the virtual environment each time you start a new terminal session.

## Start local mlflow server

Enter this into commandline after navigating to the repo or open a new terminal in VSCode for the repo and enter it there:

- On macOS and Linux:

    ```bash
    bash mlflow/start_mlflow.sh
    ```

- On Windows:

    ```bash
    .\mlflow\start_mlflow.bat
    ```

## Access local mlflow server

You can access the local mlflow server under http://localhost:5000.

## Stop local mlflow server

Just kill the terminal.

Maybe on MacOS/Linux you also need to kill the processes running on the port.

    ```bash
    lsof -ti :5000 | xargs kill
    ```

## Contact

If you run into issues feel free to contact me.
