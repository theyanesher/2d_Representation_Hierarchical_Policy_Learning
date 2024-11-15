FOR EXAMPLE USAGE OF THE CLASSES CHECK "get_handle.py" code.

This README contains some details about the classes and how to use them 



1. APPLY THE SUGMENTATION AND HOW TO USE IT => random_apply_handle_augmentation.py accepts "transforms_and_probs" when defining the class and accepts the "asset_id" and "link_name" while calling the class.

        Use transforms_and_probs is of the following form => 

                                scaling_factor = random.uniform(0.4, 2)
                                mesh_output_path = f"data/dataset/{asset_id}/handle_modified_scaled.obj"
                                handle_scaling =  Scale_Handle(scaling_factor = scaling_factor, mesh_output_path = mesh_output_path)  #DEFINE SCALING CLASS 
                                handle_shifting = Handle_Shift(handle_shift_sigma = np.array([0.2, 0.2, 0]), num_samples_to_generate = 1, max_iters = 100) # DEFINE SHIFTING CLASS 
                                transforms_and_probs = [[handle_shifting,0], [handle_scaling, 1]] # (CLASS NAME, PROBABILITY OF THIS AUGMENTATION BEING USED)
                                rand_apply= RandomApply(transforms_and_probs) # GENERATE THE RANDOM APPLY CLASS

        While calling the function pass the "asset_id" and "link_name"  are passed as parameters where =>

                                asset_id => Id of the asset that you want to Augment. 
                                link_name => Name of the link which has the handle and the door of interest.

        Returns the path to the final modified URDF file and by how much the handle has been shifted.




2. HANDLE SHIFTING AUG => handle-shifting_aug.py shifts the handles within the door.The class takes the following parameters "handle_shift_sigma = np.array([0.2, 0.2, 0])", "num_samples_to_generate = 1", "max_iters" and "link_name. Example parameters that can be passed =>

            "handle_shift_sigma = np.array([0.2, 0.2, 0])", "num_samples_to_generate = 1", "max_iters = 100"

            handle_shift_sigma decides the shift variance of the handle in each direction wrt to the present location of the handle. KEEP THE SIGMA FOR THE Z DIRECTION = 0 else there would be an exception.

            Keep num_samples_to_generate = 1 as I did not effectively implement this feature because this reduces the amount of shifts. Increasing this number would reduce your effective overall shift, so keep it 1. 

            max_iters is the maximum number of iterations for which the algorithm would try to fit the handle to the door. Post this it would say that augmentation is not possible. Hopefully we will never face this issue,  


The call function takes "asset_id", link_name and "multiaug_flag".  

                    asset_id => Id of the asset that you want to Augment. 
                    link_name => Name of the link which has the handle and the door of interest.
                    DO NOT tamper with the "multiaug_flag" without going through the code. "asset_id" basically is the object number which you want to augment.


3. HANDLE SCALING AUG => handle_scaling.py function scales the handle. The class takes the parameters "scaling_factor" and "mesh_output_path" 


        scaling_factor => The scaling factor decides by what factor the handle must be scaled. However, if the handle does not fit within the door the scaling factor is halfed so as to incorporate the handle.

        Example Use => scaling_factor = 5
        
        
        The mesh output path decides where you want to save the modified mesh .obj files and must of the following form =>

        mesh_output_path = f"data/dataset/{asset_id}/handle_modified_scaled.obj". 

        Note that there can be multiple meshes in the handle and in that case the code would save save multiple modified (scaled) meshes inside the directory (here = data/dataset/{asset_id}/) with appropriate numbers added to "handle_modified_scaled.obj". 

The call function takes "asset_id", link_name and "multiaug_flag".  

                    asset_id => Id of the asset that you want to Augment. 
                    link_name => Name of the link which has the handle and the door of interest.
                    DO NOT tamper with the "multiaug_flag" without going through the code. "asset_id" basically is the object number which you want to augment.