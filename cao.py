import pickle


if __name__ == "__main__":
    filepath = "./processed/wikipedia/wikipedia_activity_degree_s20.pkl"
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    print(data[3:5])