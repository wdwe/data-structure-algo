class KnuthMorrisPratt:
    """Knuth Morris Pratt algorithm to search for the location of the pattern in the string."""
    def __init__(self, pattern):
        """Init with a string that we wish to find in the text.
        
        This algorithm only supports ASCII strings.
        """
        self.pattern = pattern
        self.R = 256
        self._build_dfa()

    def _build_dfa(self):
        # unless overwritten later, the state go back to 0
        self.dfa = [[0] * len(self.pattern) for _ in range(self.R)]
        # if we are in state 0 and meet the same char as the first
        # charater of the pattern, we go to state 1
        self.dfa[ord(self.pattern[0])][0] = 1
        last_state = 0
        for current_state in range(1, len(self.pattern)):
            # the mismatch case
            for r in range(self.R):
                # if mismatch, we first sent pattern[1:current_state - 1]
                # through this dfa, and which arrive at last_state that we recorded
                # the state for mismatch will be the same as the state we change to
                # given the mismatch charater from the last_state
                self.dfa[r][current_state] = self.dfa[r][last_state]
            # if at this state we can match the characters, we get to move to
            # the next state
            self.dfa[ord(self.pattern[current_state])][current_state] = current_state + 1
            # update the last_state
            # we move on to the next state in the next loop
            # when programme runs, it is the same as assuming we matched all the patern[:current_state]
            # charaters. If the next character does not match (we re-run pattern[1:current_state] through the dfa),
            # given the mismatched character, we will reach a state that is the same as this last_state
            # will change to given the mismatched charater.
            # I think I am not doing a good job explaining here.
            last_state = self.dfa[ord(self.pattern[current_state])][last_state]
    
    def search(self, text):
        """Return the position of the stored pattern in the given text.
        
        -1 is returned if it is not found.
        The text must be an ASCII code string.
        """
        # current_state can be interpreted as how many chars we have matched so far
        current_state = 0
        for i, c in enumerate(text):
            current_state = self.dfa[ord(c)][current_state]
            if current_state == len(self.pattern):
                return i - current_state + 1
        return -1

class BoyerMoore:
    """Boyer Moore algorithm for substring search."""
    def __init__(self, pattern):
        """Init the instance with the pattern we wish to search.
        
        The pattern must be an ASCII string.
        """
        self.pattern = pattern
        self.R = 256
        self.position = [-1] * self.R
        for i, c in enumerate(self.pattern):
            self.position[ord(c)] = i
    
    def search(self, text):
        """Return the position of the stored pattern in the given text.
        
        -1 is returned if it is not found.
        The text must be an ASCII string.
        """
        i = 0
        while i <= len(text) - len(self.pattern):
            for j in range(len(self.pattern) -1, -1, -1):
                if text[i + j] != self.pattern[j]:
                    # if j - self.position[...] < 0:
                    # it means we will be shifting the pattern backwards
                    # which is redundant as the fact we are at this position
                    # means there is not match at any position before this
                    # therefore, we move forward by one position instead
                    i += max(1, j - self.position[ord(self.pattern[j])])
                    break
                return i
        return -1

class RabinKarp:
    """Rabin Karp algorithm for substring search.
    
    This implements the las vegas version.
    """
    def __init__(self, pattern):
        """Init with an ASCII string pattern to search for."""
        self.pattern = pattern
        self.prime = 2147483647
        self.R = 256
        # self.RM is self.R ^ (len(self.pattern) - 1) % self.prime
        # it will be used when in search where we will use honor's method
        # to compute the hash for the substrings
        self.RM = 1 
        for _ in range(len(self.pattern) - 1):
            self.RM = (self.R * self.RM) % self.prime
        self.pattern_hash = self._hash(self.pattern, len(self.pattern))

    def search(self, text):
        """Return the location of the substring in the text string.
        
        -1 is returned if the pattern is not found in the string.
        The text must be an ASCII string.
        """
        text_hash = self._hash(text, len(self.pattern))
        if text_hash == self.pattern_hash:
            if text[:len(self.pattern)] == self.pattern: # las vegas version
                return 0
        for i in range(len(self.pattern), len(text)):
            # the + self.prime is to prevent 
            text_hash = (text_hash - self.RM * ord(text[i - len(self.pattern)]) % self.prime + self.prime) % self.prime
            text_hash = (text_hash * self.R + ord(text[i])) % self.prime
            if text_hash == self.pattern_hash:
                if text[i - len(self.pattern) + 1: i + 1] == self.pattern: # las vegas version
                    return i - len(self.pattern) + 1
        return -1


    def _hash(self, string, M):
        """Computing the hash code for the first M characters of the string."""
        h = 0
        for i in range(M):
            # honor's method
            h = (self.R * h + ord(string[i])) % self.prime
        return h



if __name__ == "__main__":
    kmp = KnuthMorrisPratt("abcdddddddddddd")
    print(kmp.search("wadfbabcd"))
    bm = BoyerMoore("abc")
    print(bm.search("asdfefa;abc"))
    RK = RabinKarp("bcdef")
    print(RK.search("aaaaabcdefg"))
