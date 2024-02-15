# antBlueFinThesis


## Installation

Follow these steps to install and run the project:

1. **Download dataset from The IWC-SORP/SOOS Acoustic Trends Annotated Library**:

http://data.aad.gov.au/metadata/records/AcousticTrends_BlueFinLibrary

2. **Clone the repository**:

```bash
git clone https://github.com/Nickr234/antBlueFinThesis.git

3. **Navigate to the project directory**:

cd antBlueFinThesis

4. **Move Annotated Library into new Repository**:

cp -R /path/to/AcousticTrends_BlueFinLibrary/* /path/to/antBlueFinThesis/ ## Linux

xcopy /E /I C:\path\to\AcousticTrends_BlueFinLibrary\* C:\path\to\antBlueFinThesis\  ## Windows

5. **Set up a virtual environment** (optional but recommended):

python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

6. **Install the required packages**:

pip install -r requirements.txt


7. **Run the project**:

python runAll.py  # Replace with your main script