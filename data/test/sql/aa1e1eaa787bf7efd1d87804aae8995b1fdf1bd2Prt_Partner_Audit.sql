CREATE TABLE PRT_PARTNER_AUDIT
(
  AUDIT_ID NUMBER NOT NULL 
, COUNTRY_CODE VARCHAR2(2 BYTE) NOT NULL 
, PARTNER_ID VARCHAR2(8 BYTE) NOT NULL 
, OLD_COMPANY_TYPE VARCHAR2(3 BYTE) 
, OLD_COMPANY_NAME VARCHAR2(60 BYTE)
, OLD_DOB DATE 
, OLD_PERSON_NUMBER VARCHAR2(15 BYTE) 
, OLD_ORG_ID VARCHAR2(20 BYTE) 
, OLD_GENDER VARCHAR2(1 BYTE) 
, OLD_VAT_REG_NUMBER VARCHAR2(20 BYTE) 
, OLD_PARTNER_ADDR1 VARCHAR2(60 BYTE)
, OLD_PARTNER_ADDR2 VARCHAR2(60 BYTE)
, OLD_PARTNER_POSTAL_CODE VARCHAR2(10 BYTE)
, OLD_PARTNER_CITY VARCHAR2(60 BYTE)
, OLD_PARTNER_ADDR_SEQ VARCHAR2(5 BYTE)
, OLD_PARTNER_COUNTRY VARCHAR2(2 BYTE)
, OLD_PARTNER_ADDR_LEVEL VARCHAR2(3 BYTE)
, OLD_PARTNER_ADDR_TYPE VARCHAR2(3 BYTE)
, OLD_MODIFIED_BY VARCHAR2(50 BYTE) 
, OLD_MODIFIED_DATE TIMESTAMP(6) 
, NEW_COMPANY_TYPE VARCHAR2(3 BYTE) 
, NEW_COMPANY_NAME VARCHAR2(60 BYTE)
, NEW_DOB DATE 
, NEW_PERSON_NUMBER VARCHAR2(15 BYTE) 
, NEW_ORG_ID VARCHAR2(20 BYTE) 
, NEW_GENDER VARCHAR2(1 BYTE) 
, NEW_VAT_REG_NUMBER VARCHAR2(20 BYTE) 
, NEW_PARTNER_ADDR1 VARCHAR2(60 BYTE)
, NEW_PARTNER_ADDR2 VARCHAR2(60 BYTE)
, NEW_PARTNER_POSTAL_CODE VARCHAR2(10 BYTE)
, NEW_PARTNER_CITY VARCHAR2(60 BYTE)
, NEW_PARTNER_ADDR_SEQ VARCHAR2(5 BYTE)
, NEW_PARTNER_COUNTRY VARCHAR2(2 BYTE)
, NEW_PARTNER_ADDR_LEVEL VARCHAR2(3 BYTE)
, NEW_PARTNER_ADDR_TYPE VARCHAR2(3 BYTE)
, NEW_MODIFIED_BY VARCHAR2(50 BYTE) 
, NEW_MODIFIED_DATE TIMESTAMP(6)
, U_ID VARCHAR2(50 BYTE) 
, ACTION VARCHAR2(20 BYTE) 
, NEW_TIMESTAMP DATE 



, CONSTRAINT PRT_PARTNER_AUDIT_PK PRIMARY KEY 
  (
    AUDIT_ID 
  )
  ENABLE 
) 

PARTITION BY LIST(COUNTRY_CODE) (
    PARTITION PRT_PARTNER_AUDIT_SWEDEN VALUES ('SE'),
    PARTITION PRT_PARTNER_AUDIT_NORWAY VALUES ('NO'),
    PARTITION PRT_PARTNER_AUDIT_DENMARK VALUES ('DK'),
    PARTITION PRT_PARTNER_AUDIT_LATVIA VALUES ('LV'),
    PARTITION PRT_PARTNER_AUDIT_LITHUANIA VALUES ('LT'),
    PARTITION PRT_PARTNER_AUDIT_ESTONIA VALUES ('EE'),
    PARTITION PRT_PARTNER_AUDIT_POLAND VALUES ('PL')
);
