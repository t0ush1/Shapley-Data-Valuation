import argparse as ap
import os
import pickle as pk
import random
import numpy as np
import csv


DATASETS_DIR = "/code/TKDE-SHAP-main/sources/datasets"


def categorical(*category):
    return lambda s: category.index(s)


def onehot(*category):
    return lambda s: [1 if c == s else 0 for c in category]


SETTINGS = {
    "adult": {
        "train": "adult/adult.data",
        "test": "adult/adult.test",
        "label": "income",
        "field_names": ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"],
        "field_types": {
            "age": int,
            "workclass": categorical("Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"),
            "fnlwgt": int,
            "education": categorical("Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"),
            "education-num": int,
            "marital-status": categorical("Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"),
            "occupation": categorical("Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"),
            "relationship": categorical("Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"),
            "race": categorical("White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"),
            "sex": categorical("Female", "Male"),
            "capital-gain": int,
            "capital-loss": int,
            "hours-per-week": int,
            "native-country": categorical("United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"),
            "income": categorical(">50K", "<=50K")
        }
    },
    "adult_onehot": {
        "train": "adult/adult.data",
        "test": "adult/adult.test",
        "label": "income",
        "field_names": ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"],
        "field_types": {
            "age": int,
            "workclass": onehot("Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"),
            "fnlwgt": int,
            "education": onehot("Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"),
            "education-num": int,
            "marital-status": onehot("Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"),
            "occupation": onehot("Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"),
            "relationship": onehot("Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"),
            "race": onehot("White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"),
            "sex": onehot("Female", "Male"),
            "capital-gain": int,
            "capital-loss": int,
            "hours-per-week": int,
            "native-country": onehot("United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"),
            "income": categorical(">50K", "<=50K")
        }
    },
    "covtype": {
        "train": "covtype/covtype.data",
        "label": 55 - 4,
        "field_types": [int for i in range(55)]
    },
    "census": {
        "train": "census/census-income.data",
        "test": "census/census-income.test",
        "label": 42 - 1,
        "field_types": [
            int,
            categorical("Not in universe", "Federal government", "Local government", "Never worked", "Private", "Self-employed-incorporated", "Self-employed-not incorporated", "State government", "Without pay"),
            categorical("0", "40", "44", "2", "43", "47", "48", "1", "11", "19", "24", "25", "32", "33", "34", "35", "36", "37", "38", "39", "4", "42", "45", "5", "15", "16", "22", "29", "31", "50", "14", "17", "18", "28", "3", "30", "41", "46", "51", "12", "13", "21", "23", "26", "6", "7", "9", "49", "27", "8", "10", "20"),
            categorical("0", "12", "31", "44", "19", "32", "10", "23", "26", "28", "29", "42", "40", "34", "14", "36", "38", "2", "20", "25", "37", "41", "27", "24", "30", "43", "33", "16", "45", "17", "35", "22", "18", "39", "3", "15", "13", "46", "8", "21", "9", "4", "6", "5", "1", "11", "7"),
            categorical("Children", "7th and 8th grade", "9th grade", "10th grade", "High school graduate", "11th grade", "12th grade no diploma", "5th or 6th grade", "Less than 1st grade", "Bachelors degree(BA AB BS)", "1st 2nd 3rd or 4th grade", "Some college but no degree", "Masters degree(MA MS MEng MEd MSW MBA)", "Associates degree-occup /vocational", "Associates degree-academic program", "Doctorate degree(PhD EdD)", "Prof school degree (MD DDS DVM LLB JD)"),
            int,
            categorical("Not in universe", "High school", "College or university"),
            categorical("Never married", "Married-civilian spouse present", "Married-spouse absent", "Separated", "Divorced", "Widowed", "Married-A F spouse present"),
            categorical("Not in universe or children", "Entertainment", "Social services", "Agriculture", "Education", "Public administration", "Manufacturing-durable goods", "Manufacturing-nondurable goods", "Wholesale trade", "Retail trade", "Finance insurance and real estate", "Private household services", "Business and repair services", "Personal services except private HH", "Construction", "Medical except hospital", "Other professional services", "Transportation", "Utilities and sanitary services", "Mining", "Communications", "Hospital services", "Forestry and fisheries", "Armed Forces"),
            categorical("Not in universe", "Professional specialty", "Other service", "Farming forestry and fishing", "Sales", "Adm support including clerical", "Protective services", "Handlers equip cleaners etc ", "Precision production craft & repair", "Technicians and related support", "Machine operators assmblrs & inspctrs", "Transportation and material moving", "Executive admin and managerial", "Private household services", "Armed Forces"),
            categorical("White", "Black", "Other", "Amer Indian Aleut or Eskimo", "Asian or Pacific Islander"),
            categorical("Mexican (Mexicano)", "Mexican-American", "Puerto Rican", "Central or South American", "All other", "Other Spanish", "Chicano", "Cuban", "Do not know", "NA"),
            categorical("Female", "Male"),
            categorical("Not in universe", "No", "Yes"),
            categorical("Not in universe", "Re-entrant", "Job loser - on layoff", "New entrant", "Job leaver", "Other job loser"),
            categorical("Children or Armed Forces", "Full-time schedules", "Unemployed part- time", "Not in labor force", "Unemployed full-time", "PT for non-econ reasons usually FT", "PT for econ reasons usually PT", "PT for econ reasons usually FT"),
            int,
            int,
            int,
            categorical("Nonfiler", "Joint one under 65 & one 65+", "Joint both under 65", "Single", "Head of household", "Joint both 65+"),
            categorical("Not in universe", "South", "Northeast", "West", "Midwest", "Abroad"),
            categorical("Not in universe", "Utah", "Michigan", "North Carolina", "North Dakota", "Virginia", "Vermont", "Wyoming", "West Virginia", "Pennsylvania", "Abroad", "Oregon", "California", "Iowa", "Florida", "Arkansas", "Texas", "South Carolina", "Arizona", "Indiana", "Tennessee", "Maine", "Alaska", "Ohio", "Montana", "Nebraska", "Mississippi", "District of Columbia", "Minnesota", "Illinois", "Kentucky", "Delaware", "Colorado", "Maryland", "Wisconsin", "New Hampshire", "Nevada", "New York", "Georgia", "Oklahoma", "New Mexico", "South Dakota", "Missouri", "Kansas", "Connecticut", "Louisiana", "Alabama", "Massachusetts", "Idaho", "New Jersey"),
            categorical("Child <18 never marr not in subfamily", "Other Rel <18 never marr child of subfamily RP", "Other Rel <18 never marr not in subfamily", "Grandchild <18 never marr child of subfamily RP", "Grandchild <18 never marr not in subfamily", "Secondary individual", "In group quarters", "Child under 18 of RP of unrel subfamily", "RP of unrelated subfamily", "Spouse of householder", "Householder", "Other Rel <18 never married RP of subfamily", "Grandchild <18 never marr RP of subfamily", "Child <18 never marr RP of subfamily", "Child <18 ever marr not in subfamily", "Other Rel <18 ever marr RP of subfamily", "Child <18 ever marr RP of subfamily", "Nonfamily householder", "Child <18 spouse of subfamily RP", "Other Rel <18 spouse of subfamily RP", "Other Rel <18 ever marr not in subfamily", "Grandchild <18 ever marr not in subfamily", "Child 18+ never marr Not in a subfamily", "Grandchild 18+ never marr not in subfamily", "Child 18+ ever marr RP of subfamily", "Other Rel 18+ never marr not in subfamily", "Child 18+ never marr RP of subfamily", "Other Rel 18+ ever marr RP of subfamily", "Other Rel 18+ never marr RP of subfamily", "Other Rel 18+ spouse of subfamily RP", "Other Rel 18+ ever marr not in subfamily", "Child 18+ ever marr Not in a subfamily", "Grandchild 18+ ever marr not in subfamily", "Child 18+ spouse of subfamily RP", "Spouse of RP of unrelated subfamily", "Grandchild 18+ ever marr RP of subfamily", "Grandchild 18+ never marr RP of subfamily", "Grandchild 18+ spouse of subfamily RP"),
            categorical("Child under 18 never married", "Other relative of householder", "Nonrelative of householder", "Spouse of householder", "Householder", "Child under 18 ever married", "Group Quarters- Secondary individual", "Child 18 or older"),
            float,
            categorical("Not in universe", "Nonmover", "MSA to MSA", "NonMSA to nonMSA", "MSA to nonMSA", "NonMSA to MSA", "Abroad to MSA", "Not identifiable", "Abroad to nonMSA"),
            categorical("Not in universe", "Nonmover", "Same county", "Different county same state", "Different state same division", "Abroad", "Different region", "Different division same region"),
            categorical("Not in universe", "Nonmover", "Same county", "Different county same state", "Different state in West", "Abroad", "Different state in Midwest", "Different state in South", "Different state in Northeast"),
            categorical("Not in universe under 1 year old", "Yes", "No"),
            categorical("Not in universe", "Yes", "No"),
            int,
            categorical("Both parents present", "Neither parent present", "Mother only present", "Father only present", "Not in universe"),
            categorical("Mexico", "United-States", "Puerto-Rico", "Dominican-Republic", "Jamaica", "Cuba", "Portugal", "Nicaragua", "Peru", "Ecuador", "Guatemala", "Philippines", "Canada", "Columbia", "El-Salvador", "Japan", "England", "Trinadad&Tobago", "Honduras", "Germany", "Taiwan", "Outlying-U S (Guam USVI etc)", "India", "Vietnam", "China", "Hong Kong", "Cambodia", "France", "Laos", "Haiti", "South Korea", "Iran", "Greece", "Italy", "Poland", "Thailand", "Yugoslavia", "Holand-Netherlands", "Ireland", "Scotland", "Hungary", "Panama"),
            categorical("India", "Mexico", "United-States", "Puerto-Rico", "Dominican-Republic", "England", "Honduras", "Peru", "Guatemala", "Columbia", "El-Salvador", "Philippines", "France", "Ecuador", "Nicaragua", "Cuba", "Outlying-U S (Guam USVI etc)", "Jamaica", "South Korea", "China", "Germany", "Yugoslavia", "Canada", "Vietnam", "Japan", "Cambodia", "Ireland", "Laos", "Haiti", "Portugal", "Taiwan", "Holand-Netherlands", "Greece", "Italy", "Poland", "Thailand", "Trinadad&Tobago", "Hungary", "Panama", "Hong Kong", "Scotland", "Iran"),
            categorical("United-States", "Mexico", "Puerto-Rico", "Peru", "Canada", "South Korea", "India", "Japan", "Haiti", "El-Salvador", "Dominican-Republic", "Portugal", "Columbia", "England", "Thailand", "Cuba", "Laos", "Panama", "China", "Germany", "Vietnam", "Italy", "Honduras", "Outlying-U S (Guam USVI etc)", "Hungary", "Philippines", "Poland", "Ecuador", "Iran", "Guatemala", "Holand-Netherlands", "Taiwan", "Nicaragua", "France", "Jamaica", "Scotland", "Yugoslavia", "Hong Kong", "Trinadad&Tobago", "Greece", "Cambodia", "Ireland"),
            categorical("Native- Born in the United States", "Foreign born- Not a citizen of U S ", "Native- Born in Puerto Rico or U S Outlying", "Native- Born abroad of American Parent(s)", "Foreign born- U S citizen by naturalization"),
            categorical("0", "2", "1"),
            categorical("Not in universe", "Yes", "No"),
            categorical("0", "2", "1"),
            int,
            categorical("94", "95"),
            categorical("- 50000.", "50000+.")
        ]
    }
}


def flatten(iterator):
    for it in iterator:
        if isinstance(it, list):
            yield from flatten(it)
        else:
            yield it


def load_raw_data(datapath, label, fieldnames, fieldtypes, datasize=-1):
    is_list = fieldnames is None
    data_x, data_y = [], []
    with open(f"{DATASETS_DIR}/raw/{datapath}", "r") as fin:
        reader = csv.reader(fin, skipinitialspace=True) if is_list else \
                csv.DictReader(fin, fieldnames=fieldnames, skipinitialspace=True)
        for row in reader:
            values = row if is_list else row.values()
            items = enumerate(row) if is_list else row.items()
            if any(v == "?" for v in values):
                continue
            data_x.append(list(flatten(fieldtypes[k](v) for k, v in items if k != label)))
            data_y.append(fieldtypes[label](row[label]))
    print(f"raw datasize {len(data_x)}")
    index = list(range(len(data_x)))
    random.shuffle(index)
    index = index if datasize == -1 else index[:datasize]
    data_x = [data_x[i] for i in index]
    data_y = [data_y[i] for i in index]
    return data_x, data_y


def create_dataset(cnum, dataset, setup):
    output_dir = f"{DATASETS_DIR}/{dataset}/client_{cnum}_{setup}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    settings = SETTINGS[dataset]
    if dataset == "adult" or dataset == "adult_onehot":
        train_x, train_y = load_raw_data(settings["train"], settings["label"], settings["field_names"], settings["field_types"])
        test_x, test_y = load_raw_data(settings["test"], settings["label"], settings["field_names"], settings["field_types"])

        labels = ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"]
        index = [[] for i in range(len(labels))]
        for i, x in enumerate(train_x):
            index[x[6]].append(i)
        index = sorted(index, key=len, reverse=True)
        for i, idx in enumerate(index):
            print(f"{labels[i]}: {len(idx)}")

        client_train_x = [[train_x[i] for i in index[cid]] for cid in range(cnum)]
        client_train_y = [[train_y[i] for i in index[cid]] for cid in range(cnum)]  

        # size = len(labels) // cnum
        # client_index = [index[size * cid : size * (cid + 1)] for cid in range(cnum)]
        # client_index[-1].extend(index[size * cnum :])
        # client_index = [list(flatten(idx)) for idx in client_index]
        # client_train_x = [[train_x[i] for i in client_index[cid]] for cid in range(cnum)]
        # client_train_y = [[train_y[i] for i in client_index[cid]] for cid in range(cnum)]

    elif dataset == "covtype":
        datasize = 20000
        spilt = int(datasize * 0.8)
        data_x, data_y = load_raw_data(settings["train"], settings["label"], None, settings["field_types"], datasize)
        train_x, train_y = data_x[:spilt], data_y[:spilt]
        test_x, test_y = data_x[spilt:], data_y[spilt:]

        client_size = len(train_x) // cnum
        client_train_x = [train_x[client_size * cid : client_size * (cid + 1)] for cid in range(cnum)]
        client_train_y = [train_y[client_size * cid : client_size * (cid + 1)] for cid in range(cnum)]
    
    elif dataset == "census":
        train_x, train_y = load_raw_data(settings["train"], settings["label"], None, settings["field_types"], 15000)
        test_x, test_y = load_raw_data(settings["test"], settings["label"], None, settings["field_types"], 5000)
        
        client_size = len(train_x) // cnum
        client_train_x = [train_x[client_size * cid : client_size * (cid + 1)] for cid in range(cnum)]
        client_train_y = [train_y[client_size * cid : client_size * (cid + 1)] for cid in range(cnum)]


    
    for cid in range(cnum):
        with open(f"{output_dir}/client_trainX_{cid}.pk", "wb") as fout:
            pk.dump(np.array(client_train_x[cid]), fout)
        with open(f"{output_dir}/client_trainY_{cid}.pk", "wb") as fout:
            pk.dump(np.array(client_train_y[cid]), fout)
    
    with open(f"{output_dir}/testX.pk", "wb") as fout:
        pk.dump(np.array(test_x), fout)
    with open(f"{output_dir}/testY.pk", "wb") as fout:
        pk.dump(np.array(test_y), fout)

    for cid in range(cnum):
        print(np.array(client_train_x[cid]).shape, np.array(client_train_y[cid]).shape)
    print(np.array(test_x).shape, np.array(test_y).shape)


if __name__ == "__main__":
    parser = ap.ArgumentParser(description="Create Tabular Federated Dataset.")
    parser.add_argument("--cnum", type=int, default=3)
    parser.add_argument("--dataset", type=str, default="adult")
    parser.add_argument("--setup", type=str, default="same")
    args = parser.parse_args()
    create_dataset(args.cnum, args.dataset, args.setup)
