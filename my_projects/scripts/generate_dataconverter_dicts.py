from mmseg.utils import irl_vision_sim_classes, hots_v1_classes

# hots to hots_simple
#  dict["unique_name"] = [idxs]
hots_to_cat = {}
for class_idx, class_name in enumerate(hots_v1_classes()):
    class_cat_name = class_name.split("_")[0]
    if "juice_box" in class_name:
        class_cat_name = "juice_box"
    if "background" in class_name:
        class_cat_name = "_background_"
    if class_cat_name not in hots_to_cat.keys():
        hots_to_cat[class_cat_name] = []
    hots_to_cat[class_cat_name].append(class_idx)


for idx, (key, val) in enumerate(hots_to_cat.items()):
    
    print(f"cat name {key}, idx {idx}:\n members: {[hots_v1_classes()[item] for item in val]}")

sum_ = sum(len(val) for key, val in hots_to_cat.items())
print(f"total classes accounted for: {sum_}")

  
print("IRL CAT")  
irl_vision_to_cat = {}
for class_idx, class_name in enumerate(irl_vision_sim_classes()):
    class_cat_name = class_name.split("_")[0]
    if "juice_box" in class_name:
        class_cat_name = "juice_box"
    if "background" in class_name:
        class_cat_name = "_background_"
    if class_cat_name not in irl_vision_to_cat.keys():
        irl_vision_to_cat[class_cat_name] = []
    irl_vision_to_cat[class_cat_name].append(class_idx)


for idx, (key, val) in enumerate(irl_vision_to_cat.items()):
    
    print(f"cat name {key}, idx {idx}:\n members: {[irl_vision_sim_classes()[item] for item in val]}")

sum_ = sum(len(val) for key, val in irl_vision_to_cat.items())
print(f"total classes accounted for: {sum_}")

    

print("\nkeys not found in hots: ")
for idx_irl, (key_irl, val_irl) in enumerate(irl_vision_to_cat.items()):
    if not key_irl in hots_to_cat.keys(): 
        print(key_irl)

print(f"\n keys not found in irl_vision")
for key, val in hots_to_cat.items():
    if key not in irl_vision_to_cat.keys():
        print(key)
    

print(f"len irl,hots: {len(irl_vision_to_cat)}, {len(hots_to_cat)}")

    
#### HERE THE DICTS ARE CREATED ####
hots2hots_cat = {}
for cat_idx, (cat_name, item_idx_list) in enumerate(hots_to_cat.items()):
    for item_idx in item_idx_list:
        hots2hots_cat[item_idx] = cat_idx

irl_vision2irl_vision_cat = {}
for cat_idx, (cat_name, item_idx_list) in enumerate(irl_vision_to_cat.items()):
    for item_idx in item_idx_list:
        irl_vision2irl_vision_cat[item_idx] = cat_idx

irl_vision_cat2hots_cat = {}
for idx_irl_cat, (name_irl_cat, item_idx_list_irl) in enumerate(irl_vision_to_cat.items()):
    for idx_hots_cat, (name_hots_cat, item_idx_list_hots) in enumerate(hots_to_cat.items()):
        if name_irl_cat == name_hots_cat:
            irl_vision_cat2hots_cat[idx_irl_cat] = idx_hots_cat

hots_cat2irl_vision_cat = {
    val : key for key, val in irl_vision_cat2hots_cat.items()   
}
print("idx map:\n")
print("HOTS2HOTS_CAT = {")
for item_idx, cat_idx in hots2hots_cat.items():
    print(f"\t\t{item_idx}  :  {cat_idx},")
print("}")

print("")

print("IRL_VISION2IRL_VISION_CAT = {")
for item_idx, cat_idx in irl_vision2irl_vision_cat.items():
    print(f"\t\t{item_idx}  :  {cat_idx},")
print("}")

print("")
print("IRL_VISION_CAT2HOTS_CAT = {")
for item_idx, cat_idx in irl_vision_cat2hots_cat.items():
    print(f"\t\t{item_idx}  :  {cat_idx},")
print("}")

print("")
print("HOTS_CAT2IRL_VISION_CAT = {")
for item_idx, cat_idx in hots_cat2irl_vision_cat.items():
    print(f"\t\t{item_idx}  :  {cat_idx},")
print("}")