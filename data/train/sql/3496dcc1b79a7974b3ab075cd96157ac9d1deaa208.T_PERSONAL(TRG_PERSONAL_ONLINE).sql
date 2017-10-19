CREATE OR REPLACE TRIGGER PEADMIN.TRG_PERSONAL_ONLINE
  AFTER INSERT OR UPDATE OF C_ALLOTDATE, C_ALLOTREASON, C_DEPART, C_OFFICE, C_LINEID, C_BUSID,C_CERT2_NO,C_CERT2_NO_HEX
  ON T_PERSONAL FOR EACH ROW
BEGIN
--1.读取操作后的新记录值
--2.删除新的C_ALLOTDATE后有效的记录
 DELETE FROM T_PSN_ONLINE 
  WHERE C_BELONG = :NEW.C_BELONG
      AND C_ONLINEPSN = :NEW.C_PERSONALID
   AND C_ONDATE >= :NEW.C_ALLOTDATE;
--3.修改当前有效时间内的C_DOWNDATE为新的C_ALLOTDATE
  UPDATE T_PSN_ONLINE SET C_DOWNDATE = (:NEW.C_ALLOTDATE - 1)
  WHERE C_BELONG = :NEW.C_BELONG
      AND C_ONLINEPSN = :NEW.C_PERSONALID 
   AND C_ONDATE = (
                      SELECT MAX(C_ONDATE) 
                      FROM T_PSN_ONLINE 
                      WHERE C_BELONG = :NEW.C_BELONG 
                      AND C_ONLINEPSN = :NEW.C_PERSONALID 
                      AND C_ONDATE < :NEW.C_ALLOTDATE);
--4.插入新的状态变动记录
  INSERT INTO T_PSN_ONLINE (
  C_BELONG,
  C_ONLINEPSN ,
  C_ONDATE ,
  C_ALLOTREASON,
  C_DEPART ,
  C_OFFICE ,
  C_LINEID,
  C_BUSID,
  C_CERT2_NO,
  C_CERT2_NO_HEX,
  C_DOWNDATE,
  C_DOPSN
  )
  VALUES
    (:NEW.C_BELONG,
     :NEW.C_PERSONALID,
     :NEW.C_ALLOTDATE,
     :NEW.C_ALLOTREASON,
     :NEW.C_DEPART, 
     :NEW.C_OFFICE, 
     :NEW.C_LINEID, 
     :NEW.C_BUSID,
     :NEW.C_CERT2_NO,
     :NEW.C_CERT2_NO_HEX,
   TO_DATE('9999/12/31', 'YYYY/MM/DD'),
   :NEW.C_LASTMODIFIER);
END;
/