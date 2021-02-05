import parser
import re

from utils import inverse_dict


class BooleanModel:
    extract_tokens = re.compile(r"('(.*?)')")
    replace_tokens = r'Token(\1)'
    replace_op = (
        ('and', '&'),
        ('or', '|'),
        ('not', '-')
    )
    authorized_symbols = '()&-| '

    def __init__(self, dictionary: dict):
        self._all_docs = set(dictionary.keys())
        self._dict = inverse_dict(dictionary)
        self._dict = {k: set(v.keys()) for k, v in self._dict.items()}

        class Token:
            _dict = self._dict
            _all_docs = self._all_docs

            def __init__(self, s: [set, str]):
                if isinstance(s, set):
                    self.s = s
                else:
                    # All documents that contains token s
                    self.s = Token._dict.get(s, {})

            def __and__(self, other):
                return Token(self.s.intersection(other.s))

            def __or__(self, other):
                return Token(self.s.union(other.s))

            def __neg__(self):
                return Token(Token._all_docs.difference(self.s))

        self.Token = Token

    @staticmethod
    def _replace():
        def f(matchobj):
            s = str(f.count)
            f.d[f.count] = matchobj.group(2)
            f.count += 1
            return "'" + s + "'"

        f.count = 0
        f.d = {}
        return f

    @staticmethod
    def _replace_reverse(p):
        def f(matchobj):
            i = int(matchobj.group(2))
            s = f.d[i]
            return "'" + s + "'"

        f.d = p.d
        return f

    @staticmethod
    def _check_expression(s):
        s = re.sub(BooleanModel.extract_tokens, r'\2', s)
        try:
            eval(s)
        except SyntaxError:
            return False, "Syntax error"
        if any(x not in BooleanModel.authorized_symbols for x in set(re.findall(r'\W', s))):
            return False, "Unauthorized operators or symbols"
        return True, ""

    def eval(self, query: str):
        """evaluate boolean query. Tokens are between parenthesis and 'and', 'or', 'not' are used as logical operations.
        Example: \"('science' or 'algebra') and 'mathematics'\""""
        Token = self.Token
        query = query.lower()
        f = BooleanModel._replace()
        query = re.sub(self.extract_tokens, f, query)
        for rep, w in self.replace_op:
            query = query.replace(rep, w)
        is_correct, _ = BooleanModel._check_expression(query)
        if not is_correct:
            return None
        query = re.sub(self.extract_tokens, BooleanModel._replace_reverse(f), query)
        query = re.sub(self.extract_tokens, self.replace_tokens, query)
        return eval(query).s