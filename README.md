Biblical AI

Overview

Biblical AI is an open-source project focused on analyzing biblical texts using AI models. It aims to provide verse detection, multi-translation comparisons, and theological accuracy checks.

Features

Verse Resolution: Identifies and maps biblical references.

Cross-Reference Analysis: Detects relationships between verses.

Data Preprocessing: Prepares text data for AI processing.

Theological Context Filtering: Ensures generated outputs remain theologically sound.


Installation

1. Clone the repository:

git clone https://github.com/Ravikumarkatta/bible.git
cd bible


2. Create a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows use 'venv\Scripts\activate'


3. Install dependencies:

pip install -r requirements.txt


4. Install the package:

python setup.py install



Usage

Verse Resolver:

python scripts/verse-resolver.py --verse "John 3:16"

Cross References:

python scripts/cross-references.py

Preprocessing:

python scripts/preprocessing1.py --input data/bible.txt


Testing

Run all unit tests using:

pytest tests/

Contributing

We welcome contributions!

1. Fork the repository.


2. Create a new branch.


3. Make your changes and commit.


4. Submit a pull request.



Ethical Considerations

This tool is intended for educational use only and does not serve as a theological authority. All outputs must be interpreted within appropriate theological contexts.

License

This project is licensed under the MIT License. See the LICENSE file for details.
