import pickle
import struct
import sqlite3
from typing import Iterable, Set, Union
from itertools import cycle
from collections import defaultdict
from quick_knn.type_hints import Key


class Data(object):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def insert(self, keys: Iterable[bytes], value: Key) -> None:
        pass

    def get(self, keys: Iterable[bytes]) -> Set[Key]:
        pass

    def save(self, lsh, hasher):
        pass

    @classmethod
    def reload(cls, name):
        pass

class SQLData(Data):
    def __init__(self, name: str, *args, in_memory: bool=False, **kwargs):
        if in_memory:
            name = ':memory:'
        super().__init__(name)
        self.conn = sqlite3.connect(name, isolation_level='EXCLUSIVE')
        try:
            c = self.conn.cursor()
            c.execute('PRAGMA journal_model = WAL')
            c.execute('''CREATE TABLE lsh (key BLOB NOT NULL, value BLOB NOT NULL)''')
            c.execute('''CREATE INDEX keys ON lsh (key)''')
            self.conn.commit()
        except sqlite3.OperationalError:
            self.conn.rollback()
        finally:
            c.close()

    def insert(self, keys: Iterable[bytes], value: Key) -> None:
        try:
            c = self.conn.cursor()
            c.executemany(
                '''INSERT INTO lsh (key, value) VALUES (?, ?)''',
                zip(
                    [key + struct.pack('<I', i) for i, key in enumerate(keys)],
                    cycle([pickle.dumps(value)])
                )
            )
            self.conn.commit()
        except:
            self.conn.rollback()
        finally:
            c.close()

    def get(self, keys: Iterable[bytes]) -> Set[Key]:
        values = set()
        try:
            c = self.conn.cursor()
            c.execute(
                f'''SELECT value FROM lsh WHERE key in ({",".join(['?'] * len(keys))});''',
                [key + struct.pack('<I', i) for i, key in enumerate(keys)]
            )
            values = c.fetchall()
            values = set(pickle.loads(val[0]) for val in values)
        except:
            self.conn.rollback()
        finally:
            c.close()
        return values


class PickleData(Data):
    def __init__(self, name: str, b: int):
        super().__init__(name)
        self.tables = [defaultdict(set) for _ in range(b)]

    def insert(self, keys: Iterable[bytes], value: Key) -> None:
        for key, table in zip(keys, self.tables):
            table[key].add(value)

    def get(self, keys: Iterable[bytes]) -> Set[Key]:
        cands = set()
        for key, table in zip(keys, self.tables):
            cands.update(table[key])
        return cands

    def save(self, lsh, hasher):
        data = [self.tables, lsh, hasher]
        pickle.dump(data, open(f"{self.name}.p", "wb"))

    @classmethod
    def reload(cls, name):
        data = pickle.load(open(f"{name}.p", "rb"))
        tables, lsh, hasher = data
        data = cls(name, 1)
        data.tables = tables
        return data, lsh, hasher


def sniff(file_name: str) -> Union[SQLData, PickleData]:
    try:
        SQL_STRING = "SQLite format 3\x00"
        with open(file_name, 'rb') as f:
            header = f.read(16).decode('utf-8')
        if header == SQL_STRING:
            return SQLData
    except UnicodeDecodeError:
        pass
    return PickleData
