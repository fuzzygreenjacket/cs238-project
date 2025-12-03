from enum import Enum, auto

# this file defines a class of the cards in sushi go 

class CardType(Enum):
    TEMPURA = auto()
    SASHIMI = auto()
    DUMPLING = auto()
    MAKI_1 = auto()
    MAKI_2 = auto()
    MAKI_3 = auto()
    SALMON_NIGIRI = auto()
    SQUID_NIGIRI = auto()
    EGG_NIGIRI = auto()
    PUDDING = auto()
    WASABI = auto()
    CHOPSTICKS = auto()


CARD_NAMES = {
    CardType.TEMPURA: "Tempura",
    CardType.SASHIMI: "Sashimi",
    CardType.DUMPLING: "Dumpling",
    CardType.MAKI_1: "Maki (1)",
    CardType.MAKI_2: "Maki (2)",
    CardType.MAKI_3: "Maki (3)",
    CardType.SALMON_NIGIRI: "Salmon Nigiri",
    CardType.SQUID_NIGIRI: "Squid Nigiri",
    CardType.EGG_NIGIRI: "Egg Nigiri",
    CardType.PUDDING: "Pudding",
    CardType.WASABI: "Wasabi",
    CardType.CHOPSTICKS: "Chopsticks",
}
