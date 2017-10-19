MERGE INTO TAG EXISTING_TAG USING
  (SELECT DISTINCT POLLUTANTGROUP_KEY "KEY",
                POLLUTANTGROUP "NAME",
                POLLUTANTGROUP DESCRIPTION,
                (SELECT ID FROM TAGGROUP WHERE TAGGROUP.KEY = 'POLLUTANTGROUP') TAGGROUP,
                'POLLUTANTGROUP' TAGGROUP_KEY
    FROM CSX_REFLIST UNION
    SELECT DISTINCT POLLUTANT_KEY "KEY",
                POLLUTANT_NAME "NAME",
                POLLUTANT_DESCRIPTION DESCRIPTION,
                (SELECT ID FROM TAGGROUP WHERE TAGGROUP.KEY = 'POLLUTANT') TAGGROUP,
                'POLLUTANT' TAGGROUP_KEY
    FROM CSX_REFLIST) NEW_TAG 
ON (EXISTING_TAG.KEY = NEW_TAG.KEY
   AND EXISTING_TAG.TAGGROUP = NEW_TAG.TAGGROUP) 
    WHEN NOT matched THEN insert(KEY, NAME, DESCRIPTION, TAGGROUP, TAGGROUP_KEY)
    VALUES(NEW_TAG.KEY, NEW_TAG.NAME,
                        NEW_TAG.DESCRIPTION,
                        NEW_TAG.TAGGROUP,
                        NEW_TAG.TAGGROUP_KEY);