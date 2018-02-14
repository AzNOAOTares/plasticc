import sys
import pymysql
import getpass

_SQL_FLAVOR = 'mysql'
_MYSQL_CONFIG = {'host':'dlgreenmysqlv.stsci.edu',
                'port':43306,
                'user':'antares',
                'db':'yse',
                'password':None}


def get_sql_password():
    """
    Prompts the user for a mysql password 
    Only accepts input from specified users
    """
    user = None
    try:
        user = getpass.getuser()
        password = getpass.getpass(prompt='Enter MySQL password: ', stream=sys.stderr)
        if user not in ['gnarayan', 'dmuthukrishna']:
            message = 'Unauthorized user {}'.format(user)
            raise RuntimeError(message)
        _MYSQL_CONFIG['password'] = password
    except Exception as e:
        message = '{}\nCould not get password from user {}'.format(e, user)
        raise RuntimeError(message)


def check_sql_db_for_table(table_name):
    """
    Check if a MySQL Table exists for this data_release
    """
    query = 'show tables'
    result = exec_sql_query(query)
    existing_tables = [table[0] for table in result]
    if table_name in existing_tables:
        return True 
    else:
        return False


def drop_sql_table_from_db(table_name):
    """
    Drops a MySQL Table from the table (assumes it exists)
    """
    query = 'drop table {}'.format(table_name)
    result = exec_sql_query(query)
    return result


def get_index_table_name_for_release(data_release):
    """
    Some datareleases are defined purely as ints which make invalid MySQL table names
    Fix that here by always prefixing release_ to the data_release name to make the MySQL table_name
    """
    table_name = 'release_{}'.format(data_release)
    return table_name


def create_sql_index_table_for_release(data_release, redo=False):
    """
    Creates a MySQL Table to hold useful data from HEAD.fits files from the
    PLASTICC sim. Checks if the table exists already. Drop if redo.
    Returns a table_name
    """
    table_name = get_index_table_name_for_release(data_release)
    result = check_sql_db_for_table(table_name)
    if result:
        print("Table {} exists.".format(table_name))
        if redo:
            print("Clobbering table {}.".format(table_name))
            drop_sql_table_from_db(table_name)
        else:
            return table_name

    query = 'CREATE TABLE {} (objid TINYTEXT, ptrobs_min BIGINT UNSIGNED, ptrobs_max BIGINT UNSIGNED, mwebv FLOAT, mwebv_err FLOAT, hostgal_photoz FLOAT, hostgal_photoz_err FLOAT, sntype SMALLINT UNSIGNED, peakmjd FLOAT)'.format(table_name) 
    result =  exec_sql_query(query)
    print("Created Table {}.".format(table_name))
    return table_name


def write_rows_to_index_table(index_entries, table_name):
    """
    Write rows to an index table
    Returns number of rows written
    """
    con = get_mysql_connection()
    query = 'INSERT INTO {} VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)'.format(table_name)
    cursor = con.cursor()
    number_of_rows = cursor.executemany(query, index_entries)
    con.commit()
    con.close()
    return number_of_rows


def get_mysql_connection():
    """
    Get a  MySQL connection object. The config variable _MYSQL_CONFIG defines
    the context/parameters of the connection. If config does not include a
    MySQL password, the user is prompted for it.
    Returns a MySQL connection object.
    """
    password = _MYSQL_CONFIG.get('password')
    if password is None:
        get_sql_password()
    con = pymysql.connect(**_MYSQL_CONFIG)
    return con


def exec_sql_query(query):
    """
    Executes a supplied MySQL query. The context of the query is defined by
    _MYSQL_CONFIG and the connection object returned by get_mysql_connection() 
    Returns the result of the query (if any)
    """
    result = None
    con = get_mysql_connection()
    try:
        cursor = con.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
    except Exception as e:
        message = '{}\nFailed to execute query\n{}'.format(e, query)
        raise RuntimeError(message)
    finally:
        con.close()
    return result
