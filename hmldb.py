# coding: utf-8
import sqlite3
import os

class HmlDB(object):

    def __init__(self, dbname):
        """
        Args:
        """
        #self.conn = sqlite3.connect(os.path.join(os.environ.get('HOME'), 'app-root/repo/data/'+dbname))

        self.conn = sqlite3.connect(dbname)
        self.cur = self.conn.cursor()
        self.cur.execute('CREATE TABLE IF NOT EXISTS words (lemma TEXT, token TEXT, msd TEXT, PRIMARY KEY (lemma, token, msd))')
        self.cur.execute('CREATE TABLE IF NOT EXISTS nowords (token TEXT, PRIMARY KEY (token))')

    def __del__(self):
        self.cur.close()
        self.conn.close()

    def count_tokens(self):
        """
        Returns count of tokens
        """
        self.cur.execute('SELECT COUNT( DISTINCT token) AS token_count FROM words')
        return self.cur.fetchone()[0]

    def count_lemmas(self, msd):
        """
        Returns count of distinct lemmas
        """
        self.cur.execute('SELECT COUNT(DISTINCT lemma) AS lemma_count FROM words where msd like ?',(msd, ))
        result = self.cur.fetchone()[0]
        return result
    def count_lemmasAndTokens(self):
        """
        Returns count of distinct lemmas
        """
        self.cur.execute('SELECT COUNT(*) FROM (SELECT DISTINCT lemma,token FROM words) AS lemmaToken_count')
        return self.cur.fetchone()[0]
    def count_tokens_by_msd(self, msd):
        """
        Returns count of distinct lemmas
        """
        self.cur.execute('SELECT COUNT(token) FROM words WHERE msd like ?', (msd, ))
        return self.cur.fetchone()[0]
    def select_all(self):
        """
        Returns all triples
        """
        self.cur.execute('SELECT DISTINCT * FROM words')
        return self.cur.fetchall()
    def select_all_lemmaTokenMsd(self):
        """
        Returns all pairs
        """
        self.cur.execute('SELECT distinct lemma,token,msd FROM words')
        return self.cur.fetchall()

    def select_lemmas(self):
        """
        Returns all lemmas
        """
        self.cur.execute('SELECT DISTINCT lemma FROM words')
        return [row[0] for row in self.cur.fetchall()]


    def select_by_lemma(self, lemma):
        """
        Returns all triples having given lemma
        """
        #ako koristimo '=' umjesto LIKE znak % nije prepoznat
        self.cur.execute('SELECT * FROM words WHERE lemma LIKE ?', (lemma, ))
        return self.cur.fetchall()

    def select_by_lemmas(self, lemmas, group_by = True):
        """
        Returns all triples having any of given lemmas
        and optionally groups them by the lemma
        """
        self.cur.execute('SELECT * FROM words WHERE lemma IN (%s)' % ', '.join(map(repr, lemmas)))
        triples = self.cur.fetchall()
        if group_by:
            group = {}
            for triple in triples:
                lemma = triple[0]
                group.setdefault(lemma, []).append(triple)
            for lemma in lemmas:
                group.setdefault(lemma, [])
            return group
        else:
            return triples

    def select_by_token(self, token):
        """
        Returns all triples having given token
        """
        self.cur.execute('SELECT msd FROM words WHERE token LIKE ?', (token, ))
        return self.cur.fetchall()[0][0][0]

    def select_by_tokens(self, tokens, group_by = True):
        """
        Returns all triples having any of given tokens
        and optionally groups them by the token
        """
        self.cur.execute('SELECT * FROM words WHERE token IN (%s) COLLATE NOCASE' % ', '.join(map(repr, tokens)))
        triples = self.cur.fetchall()
        if group_by:
            group = {}
            for triple in triples:
                token = triple[1].lower()
                group.setdefault(token, []).append(triple)
            for token in tokens:
                group.setdefault(token, [])
            return group
        else:
            return triples

    def select_by_msd(self, msd):
        """
        Returns lemma,token having msd like given msd
        """
        self.cur.execute('SELECT DISTINCT(lemma,token) FROM words WHERE msd LIKE ?', (msd, ))
        return self.cur.fetchall()
    def select_token_by_msd(self, msd):
        """
        Returns all tokens having msd like given msd
        """
        #msd = msd.replace('-','_').strip('%') + '%'
        self.cur.execute('SELECT DISTINCT token from words where msd like ? ', (msd,) )
        return self.cur.fetchall()

    def select_any(self, lemma = None, token = None, msd = None):
        """select_any(lemma = None, tokk)
        Returns triples having given lemma, token and/or msd.
        Useful for filtering.
        """
        params, where = [], []
        if lemma:
            params.append(lemma)
            where.append('lemma LIKE ?')
        if token:
            params.append(token)
            where.append('token LIKE ?')
        if msd:
            tmp = msd.replace('-','_').strip('%') + '%'
            params.append(tmp)
            where.append('msd LIKE ?')

        if not params:
            self.cur.execute('SELECT * FROM words')
        else:
            where = ' AND '.join(where)
            self.cur.execute('SELECT * FROM words WHERE ' + where, params)
        return self.cur.fetchall()



if __name__ == '__main__':
    hmldb = HmlDB('../hml.db')

