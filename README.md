## Instructions


Steps to run the script

### Running script for the first time
this sections shows how to create virtual environment for
our script and how to install dependencies
1. Open folder in terminal
```bash
cd <root_folder_of_project>/
```
2. Create virtual env
```bash
python3 -m venv venv/
```
3. Open virtual env
```bash
source venv/bin/activate
```
4. Install required dependencies
```bash
pip install -r requirements.txt
```
you can check if dependencies were installed by running next command,it should print list with installed dependencies
```bash
pip list
```

5. Close virtual env
```bash
deactivate
```

## Execute scripts

1.open virtual env
```bash
source venv/bin/activate
```
2. Running the script

	2.1.  For **wines dataset** execute
	```bash
	python3 main_wines.py
	```

	2.2. For **hypothyroid dataset** execute
	```bash
	python3 main_hypo.py
	```
 	2.3 For **cmc dataset** execute
	```bash
	python3 main_cmc.py
	```

3. Close virtual env
```bash
deactivate
```

