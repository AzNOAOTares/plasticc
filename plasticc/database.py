import sys
import pymysql
import getpass

# a few globals
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


def check_sql_table_for_release(data_release):
    """
    Check if a MySQL Table exists for this data_release
    """
    query = 'show tables'
    result = exec_sql_query(query)
    print(result)


def create_sql_table_for_release(data_release):
    """
    Creates a MySQL Table to hold useful data from HEAD.fits files from the
    PLASTICC sim
    """
    query = 'CREATE TABLE {} (id varchar(50), ptrobs_min int, ptrobs_max int, mwebv float, mwebv_err float,\
            hostgal_photoz float, hostgal_photoz_err float, sntype int, peakmjd float)'.format(data_release) 
    return exec_sql_query(query)


def exec_sql_query(query):

    password = _MYSQL_CONFIG.get('password')
    if password is None:
        get_sql_password()

    con = pymysql.connect(**_MYSQL_CONFIG)
    result = None
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
