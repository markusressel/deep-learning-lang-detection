/* Formatted on 8/23/2012 1:52:22 PM (QP5 v5.215.12089.38647) */
CREATE TABLE PROTECT.INCIDENT_LOG
AS
   SELECT *
     FROM PROTECT.INCIDENT
    WHERE 1 = 2;

ALTER TABLE PROTECT.INCIDENT_LOG ADD (LOGTIME TIMESTAMP(6));

CREATE OR REPLACE TRIGGER PROTECT.INCIDENT_UPDATE
   AFTER UPDATE OF INCIDENTSEVERITYID, ISDELETED, MESSAGETYPE, BLOCKEDSTATUS, DISCOVERITEMID, DATAOWNERID, DATAOWNEREMAILID, ISBLOCKEDSTATUSSUPERSEDED
   ON PROTECT.INCIDENT
   REFERENCING NEW AS NEW OLD AS OLD
   FOR EACH ROW
DECLARE
BEGIN
   INSERT INTO PROTECT.INCIDENT_LOG
      (SELECT :NEW.INCIDENTID,
              :NEW.MESSAGEID,
              :NEW.POLICYID,
              :NEW.POLICYVERSION,
              :NEW.INCIDENTSTATUSID,
              :NEW.VIOLATIONCOUNT,
              :NEW.DETECTIONDATE,
              :NEW.POLICYGROUPID,
              :NEW.CUSTOMATTRIBUTESRECORDID,
              :NEW.ISDELETED,
              :NEW.BLOCKEDSTATUS,
              :NEW.INCIDENTSEVERITYID,
              :NEW.MESSAGETYPE,
              :NEW.DISCOVERITEMID,
              :NEW.DISCOVERMILLISSINCEFIRSTSEEN,
              :NEW.CREATIONDATE,
              :NEW.DATAOWNERID,
              :NEW.DATAOWNEREMAILID,
              :NEW.ISBLOCKEDSTATUSSUPERSEDED,
              CURRENT_TIMESTAMP
         FROM DUAL);
EXCEPTION
   WHEN OTHERS
   THEN
      NULL;
END INCIDENT_UPDATE;