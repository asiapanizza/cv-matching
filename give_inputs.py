def give_inputs():
        satisfied = False
        while not satisfied:
            # 1. Choose query object
            answered = False 
            while not answered:
                print("What are you querying with?")
                query_object = input("Write [J] for 'job' or [C] for cv: ").lower()
                if query_object == "j":
                    query_with_cv = False
                    answered = True
                elif query_object == "c":
                    query_with_cv = True
                    answered = True
                else:
                    answered = False
            # 2. Choose file type
            print("What type of file is your cv?")
            file_chosen = False
            while not file_chosen:
                file_question = input("Choose [TXT] or [PDF]: ").lower()
                if file_question == "txt":
                    file_type = "txt"
                    file_source = "new_texts"
                    file_chosen = True
                elif file_question == "pdf":
                    file_type = "pdf"
                    file_source = "linkedin_pdf"
                    file_chosen = True
                else:
                    file_chosen = False
            # 3. Path Input and Validation
            tell_path = False
            while not tell_path:
                path = input("Please, indicate your file path: ") # Added input()
                if path.lower().endswith(file_type): # Safer way to check extension
                    tell_path = True
                else:
                    print(f"Error: Path does not end in .{file_type}. Try again.")

            # 4. Confirmation
            print(f"\n--- Confirm Choices ---")
            print(f"Path: {path}\nType: {file_type}\nSource: {file_source}")
            
            if input("Yes [Y] or no [N]?: ").lower() == "y":
                satisfied = True

        new_file = {"path": path, "source": file_source, "type": file_type}
        return new_file, query_with_cv

def select_integer():
    print("How many texts do you want ot retrieve?")
    while True:
        number = input("Insert integer value")
        try:
            return int(number)
        except ValueError:
            print("That's not an integer, try again")
