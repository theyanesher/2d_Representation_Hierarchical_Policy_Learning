import torch

siglip_yufei = torch.load("./siglip_text_features.pt")
# siglip_cy_pick_and_place = torch.load("./siglip_text_features_pick_and_place.pt")
siglip_cy_including_grasping_and_lifting = torch.load("./siglip_text_features_grasp_and_lift.pt")

new_siglip = siglip_cy_including_grasping_and_lifting.clone()
new_siglip[1] = siglip_yufei[1]
new_siglip[2] = siglip_yufei[2]

dict = {
    "keys": [
        "open the storage furniture",
        "open the bucket",
        "open the faucet",
        "open the folding chair",
        "open the laptop",
        "open the stapler",
        "open the toilet",
        "close the storage furniture",
        "close the folding chair",
        "close the laptop",
        "close the stapler",
        "close the toilet",
        "grasp and lift the object",
        "put object A on top of object B",
        "put object A inside object B",
        "grasp any object",
    ],
    
    "values": new_siglip
}

torch.save(dict, "siglip_text_features_w_pick_and_place_w_grasp_and_lift.pt")
