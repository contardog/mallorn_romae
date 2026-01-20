from romae.utils import get_encoder_size

def override_encoder_size(size: str):
    """Get the parameters of a specific RoMAE model encoder size.
    """
    try:
        ## Check RoMAE's code for the already predefined sizes with larger dimension of embeddings 
        return get_encoder_size(size)
    except:
        
        match size:
            case "tiny-midshallow10":
                return {
                    "d_model": 180,
                    "nhead": 3,
                    "depth": 10 #Use to be 2
                }
            case "tiny-midshallow8":
                return {
                    "d_model": 180,
                    "nhead": 3,
                    "depth": 10 #Use to be 2
                }
            case "tinyer-midshallow":
                return {
                    "d_model": 120,
                    "nhead": 3,
                    "depth": 10 #Use to be 2
                }
            case "very-tiny-shallow":
                return {
                    "d_model": 60,
                    "nhead": 3,
                    "depth": 6 #Use to be 2
                }
            case "very-tiny":
                return {
                    "d_model": 60,
                    "nhead": 3,
                    "depth": 12
                }
            case "super-tiny":
                return {
                    "d_model": 20,
                    "nhead": 3,
                    "depth": 12
                }
            case "super-tiny-shallow":
                return {
                    "d_model": 20,
                    "nhead": 3,
                    "depth": 2
                }
            
            case "brutally-mini":
                return {
                    "d_model": 8,
                    "nhead": 3,
                    "depth": 12
                }
            
            # case "brutally-mini-shallow":
            #     return {
            #         "d_model": 8,
            #         "nhead": 3,
            #         "depth": 2
            #     }
            case "2d": # So, this *runs* but it is probably not a good idea. Could we get smaller embeddings as we go tho?
                return {
                    "d_model": 2,
                    "nhead": 3,
                    "depth": 12
                }
            # case "2d-shallow":
            #     return {
            #         "d_model": 2,
            #         "nhead": 3,
            #         "depth": 2
            #     }
            case _:
                raise ValueError(f"Unknown encoder size: {size}")