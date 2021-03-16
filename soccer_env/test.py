import numpy as np
from gym import spaces

 action_mask=spaces.Box(0,1,shape=(14,),dtype=np.float32)
 print(action_mask)