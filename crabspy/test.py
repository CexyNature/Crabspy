import methods

# class CrabNames(object):
#
#     instances = []
#
#     def __init__(self, crab_name, crab_start_position, crab_species, sex, crab_handedness):
#         self.__class__.instances.append(self)
#         self.crab_name = crab_name
#         self.start_position = crab_start_position
#         self.species = crab_species
#         self.sex = sex
#         self.handedness = crab_handedness
#
#     def get_crab_names_test(info_video):
#         if isinstance(info_video, dict):
#             filename = "results/" + info_video.get("name_video", "")
#         else:
#             filename = info_video
#         file = open(filename, "rb")
#         temp_list = pickle.load(file)
#         temp_names_list = []
#         for i in temp_list:
#             temp_names_list.append(i.crab_name)
#
#         return temp_names_list
#
#     def save_crab_names_test(self, info_video):
#         if name in CrabNames.get_crab_names_test("results/" + video_name):
#             pass
#         else:
#             filename = "results/" + info_video.get("name_video", "")
#             file = open(filename, "wb")
#             pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
#             file.close()


video_name = "GP010016"
name = "testing"

# try:
#     name = video_name + "_" + str(input("* Please enter name for this individual: "))
#     # species = str(input("* Please enter species name for this individual: "))
#     # sex = str(input("* Please enter sex for this individual: "))
#     # handedness = str(input(" *Please enter handedness for this individual: "))
# except ValueError:
#     print("Error in input. Using pre-defined information")
#     name = video_name + "_" + methods.random_name()


print("I have given you a name. Your name is ", name)

info = [methods.CompileInformation("name_video", video_name),
        methods.CompileInformation("local_creation", "local_creation"),
        methods.CompileInformation("creation", "creation"),
        methods.CompileInformation("length_vid", "length_vid"),
        methods.CompileInformation("fps", "fps"),
        methods.CompileInformation("vid_duration", "vid_duration"),
        methods.CompileInformation("target_frame", "target_frame"),
        methods.CompileInformation("side", "side"),
        methods.CompileInformation("conversion", "conversion"),
        methods.CompileInformation("tracker", "str(tracker)"),
        methods.CompileInformation("Crab_ID", name)]

info_video = {}
for i in info:
    info_video[i.name] = i.value

# if os.path.isfile("results/" + video_name):
#     try:
#         # database = methods.CrabNames.open_crab_names(info_video)
#         # target_name = video_name + "_" + name
#         print("I am looking this crab name in the database: ", name)
#         # res = methods.CrabNames.get_crab_names("results/" + video_name)
#         # print(res)
#
#         if name in CrabNames.get_crab_names_test("results/" + video_name):
#             print("Yes, file exists and crab name found")
#
#             database = methods.CrabNames.open_crab_names(info_video)
#             for i in database:
#                 if i.crab_name == name:
#                     head_true = False
#                     sex = i.sex
#                     species = i.species
#                     handedness = i.handedness
#                     # print("This is a {} from species {} and {} handed".format(sex, species, handedness))
#                 else:
#                     pass
#             # head_true = False
#             # species = "Hola0001"
#             # sex = "Hola0001"
#             # handedness = "Hola0001"
#
#         else:
#             print("Crab name not found in database")
#             species = str(input("* Please enter species name for this individual: "))
#             sex = str(input("* Please enter sex for this individual: "))
#             handedness = str(input("*Please enter handedness for this individual: "))
#             crab_id = CrabNames(name, str("hola"), species, sex, handedness)
#             print(crab_id)
#             head_true = True
#             # print("No, file exists and crab name was not found")
#             # methods.data_writer(args["video"], info_video, head_true)
#     except (TypeError, RuntimeError):
#         pass
# # if not os.path.isfile("results" + video_name):
# else:
#     print(video_name, "A database for this video was not found. Creating a new database")
#     head_true = True
#     species = str(input("* Please enter species name for this individual: "))
#     sex = str(input("* Please enter sex for this individual: "))
#     handedness = str(input(" *Please enter handedness for this individual: "))
#     crab_id = CrabNames(name, str("hola"), species, sex, handedness)
#     print(crab_id)
#     # methods.data_writer(args["video"], info_video, head_true)
# #
# CrabNames.save_crab_names_test(CrabNames.instances, info_video)
# database = methods.CrabNames.get_crab_names("results/" + video_name)
# print(database)

database = methods.CrabNames.open_crab_names(info_video)

# if not any (i.crab_name == "test4" for i in database):
#     print("Yes")
#     # for i in database:
#     #     print("Yes")
#     #     print(i.sex)
# else:
#     print("Not found")

# print(list(i.crab_name())[list(i.values()).index("test4")])
# map(lambda d: d["crab_name"], database)

# for i in database:
#     if i.crab_name == (info_video.get("name_video") + name):
#         sex = i.sex
#         species = i.species
#         handedness = i.handedness
# #         print("This is a {} from species {} and {} handed".format(sex, species, handedness))
#     else:
#         pass
#         print("Nope")

# print("#################### End ##########################\n")
# for i in database:
#     print(i)
    # print("New line =======>\n")
    # print(i.crab_name)
    # print(i.species)
    # print(i.sex)
    # print(i.handedness)
# print("################# NOT YET ###############################")
# res = methods.CrabNames.get_crab_names("results/" + video_name)
# print(res)

# for i in database:
#     if not any (i.crab_name == "test4"):
#         print(i.sex)
#         print(i.species)
#         print(i.handedness)
#     else:
#         print("Not on list")

# methods.CrabNames.print_crab_names(info_video)
print(database)



# mylist = get_crab_names_test(info_video)
# print(mylist)
