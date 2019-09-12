import pickle
import methods

prune = True
video_name = "GP010016_fast"
file = "results/" + video_name

if prune:
    database = methods.CrabNames.open_crab_names(file)
    print("Current database \n")
    print(database)
    try:
        name = str(input("* Please enter name for entry you want to remove: "))
        individual_name = video_name + "_" + name
        print("You are about to delete entry {} from database".format(individual_name))
        answer = input("Confirm this action (type 'True') or cancel (type 'False'): ")
        if answer == "True":
            for i in database:
                if i.crab_name == individual_name:

                    res = [i for i in database if not (i.crab_name == individual_name)]
                    print("This will be the result:\n")
                    print(res)
                    # del database[i]

                    with open(file, "wb") as f:
                        pickle.dump(res, f)
                    break

        elif answer == "False":
            print("Operation halted, entry was not deleted")
            pass
        else:
            print("Unexpected answer. Operation halted")
            pass

    except ValueError:
        print("Error in input. Database was not pruned")

    # print("Current database contains:\n")
    # print(database)

else:
    database = methods.CrabNames.open_crab_names(file)
    print(database)
