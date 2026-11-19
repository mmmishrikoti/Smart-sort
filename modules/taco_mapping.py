"""
Shared category mapping from TACO's 60 fine-grained categories to SortSmart's 6 target classes:
cardboard, glass, metal, paper, plastic, trash.
"""

# Mapping by TACO Category ID (0 to 59)
TACO_ID_TO_SORTSMART = {
    # Metal
    0: "metal",       # Aluminium foil
    2: "metal",       # Aluminium blister pack
    8: "metal",       # Metal bottle cap
    10: "metal",      # Food Can
    11: "metal",      # Aerosol
    12: "metal",      # Drink can
    28: "metal",      # Metal lid
    50: "metal",      # Pop tab
    52: "metal",      # Scrap metal

    # Glass
    6: "glass",       # Glass bottle
    9: "glass",       # Broken glass
    23: "glass",      # Glass cup
    26: "glass",      # Glass jar

    # Cardboard
    13: "cardboard",  # Toilet tube
    14: "cardboard",  # Other carton
    15: "cardboard",  # Egg carton
    16: "cardboard",  # Drink carton (Tetra Pak)
    17: "cardboard",  # Corrugated carton
    18: "cardboard",  # Meal carton
    19: "cardboard",  # Pizza box

    # Paper
    20: "paper",      # Paper cup
    30: "paper",      # Magazine paper
    31: "paper",      # Tissues
    32: "paper",      # Wrapping paper
    33: "paper",      # Normal paper
    34: "paper",      # Paper bag
    35: "paper",      # Plastified paper bag
    56: "paper",      # Paper straw

    # Plastic
    4: "plastic",     # Other plastic bottle
    5: "plastic",     # Clear plastic bottle
    7: "plastic",     # Plastic bottle cap
    21: "plastic",    # Disposable plastic cup
    22: "plastic",    # Foam cup
    24: "plastic",    # Other plastic cup
    27: "plastic",    # Plastic lid
    29: "plastic",    # Other plastic
    36: "plastic",    # Plastic film
    37: "plastic",    # Six pack rings
    38: "plastic",    # Garbage bag
    39: "plastic",    # Other plastic wrapper
    40: "plastic",    # Single-use carrier bag
    41: "plastic",    # Polypropylene bag
    42: "plastic",    # Crisp packet
    43: "plastic",    # Spread tub
    44: "plastic",    # Tupperware
    45: "plastic",    # Disposable food container
    46: "plastic",    # Foam food container
    47: "plastic",    # Other plastic container
    48: "plastic",    # Plastic glooves
    49: "plastic",    # Plastic utensils
    54: "plastic",    # Squeezable tube
    55: "plastic",    # Plastic straw
    57: "plastic",    # Styrofoam piece

    # Trash / Other
    1: "trash",       # Battery
    3: "trash",       # Carded blister pack
    25: "trash",      # Food waste
    51: "trash",      # Rope & strings
    53: "trash",      # Shoe
    58: "trash",      # Unlabeled litter
    59: "trash",      # Cigarette
}

SORTSMART_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(SORTSMART_CLASSES)}
