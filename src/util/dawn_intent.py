from typing import List
INTENT_LIST: List[str] = ['greetings', 'thanks', 'bye', 'news', \
  'weather', 'worldCup', 'pkmGo', 'help', 'compliment']
N_CLASS: int = 9


def map_intent_to_number(intent: str) -> int:
    ''' Convert intent string to number indexes
    Args:
            intent (str): string intent
    Returns:
            numeric nidex representing the intent
    '''
    return INTENT_LIST.index(intent)


def map_number_to_intent(index: int) -> str:
    '''
    Args:
            n (int): numeric index representing the intent
    Returns:
            String that represents the intent
    '''
    return INTENT_LIST[index]
