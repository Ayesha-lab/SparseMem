# Iterative purning using UnitMem

### main.py:

Defines following settings based on arguments received from args_parser.

1. Model to use
2. Pruning method to use (random, low_unitmem, high_unitmem, mag_pruning)

An experiment file will be created for the current run. This will contain for each level:

    - Pruned and retrained model checkpoint files 
    - All neurons pruned as pruned_indices.json
           {level_n:    {layer_1: [neuron_1, neuron_6]
                         layer_2: [neuron_3, neuron_9, neuron_0]
                        ...
                         layer_n: []
                        }
                    ...
           }
            
    - UnitMem Values as Avg_UnitMems_perlevel.json
             {level_n:      {layer_1:   [[UM_neuron_1, img_idx]
                                        [UM_neuron_2, img_idx]
                                        ...
                                        [UM_neuron_n, img_idx]]}

                            {layer_2:   [[UM_neuron_1, img_idx]
                                        [UM_neuron_2, img_idx]
                                        ...]}
                        ...}
    
    - All activations by each image for each neuron as All_activations.json

3. Create sparsity list:

    * For UnitMem pruning: A simple squence is created [5.0, 9.75, 14.26, ... ]

    * For Magnitude pruning: A list of num_of_weights to prune is created based on how many weights (respective to neurons) were pruned when iteratively pruning in UnitMem pruning

4. For args.lvl 
                
                -1 : a model is pretrained and stored for given file name

                 0 :  a pretrained model is loaded and passed to pruner directly
            
5. For n iterations:    
    - prune model
    - update list of all neurons pruned
    - retrain model

### pruners/prunitmem2.py:
contains two functions: get_unitmem, prune_by_unitmem

1. **get_unitmem:**

    * UM_batch = batch size to use when computing unitmem

    * Augmentation:
        * Custom dataset class:
        for the same image, return 10 random augmented versions

        * Harder augmentations (than those used during training) are applied to an unaugmented dataset. We don't simply augment the training set. This ensures that augmentations are not applied twice. 

    *   Use a pre-made list of a 1000 indices to use for computing UnitMem
    * Use subset in dataloader and pass to augmentation class
    * For all ReLU layers
        * For each image_batch (a batch containing 10 augmentations of the image):
            * create a hook at the layer
            * get all activations, take mean and save them in "act"

    * On "act" (ndarray) perform UnitMem computation

2. **prune_by_unitmem:**  

    * Create pruning instance using customized structured pruner from pytorch
    * Set sparsity based on highest/lowest/hybrid
    * get_neurons_to_prune:
        * create empty dictionary for tracking neurons pruned if level == 0 
        * from current unitmem_values (UnitMem for all neurons in network)
            * remove neurons that were previously pruned
        * get number_of_neurons to prune based on x% 
        * create new dictionary containing indices that will be pruned
        * update dictionary which tracks all neurons that have been pruned
        * return updated "prev_pruned_ind". And "new_indices": neurons to prune next
    
    * Save all data
    * For each layer, pass relevant indices to pruner instance and apply mask
    * Print to make sure mask is applied (for level > 0, masks are stored in a container)

    * Return updated indices dictionary

### pruners/prune_rand_struct.py:

1. flatten_dict:
    * takes a dictionary and flattens it to a list of tuples
    * each tuple contains (layer_name, neuron_idx)
1. prune_random_neurons:
    * run get_unitmem to compute unitmem each level
    * create empty dictionary for tracking neurons pruned if level == 0 
    * from current unitmem_values (UnitMem for all neurons in network)
        * remove neurons that were previously pruned
        * compute how many neurons to prune next, save to "total_to_prune"
    * Flatten dictionary, shuffle, grab "total_to_prune", put back into dictionary    
    * Save all info
    * For each layer, pass "num_neurons" random_structured pruner from pytorch to prune







    




