--------------------------------------------------------
--  File created - Wednesday-May-28-2014   
--------------------------------------------------------
--------------------------------------------------------
--  DDL for Table HI_ACLEVEL_PREV
--------------------------------------------------------

  CREATE TABLE HI_ACLEVEL_PREV 
   (	AMC_CODE VARCHAR2(100 BYTE), 
	SCHEME VARCHAR2(100 BYTE), 
	CONTROL_AC VARCHAR2(100 BYTE), 
	RECTYPE VARCHAR2(1 BYTE), 
	CURRNO NUMBER(5,0), 
	LEVELS NUMBER(5,0), 
	PRT_LEVEL NUMBER(5,0), 
	NAME VARCHAR2(500 BYTE), 
	LEVEL_1 VARCHAR2(100 BYTE), 
	LEVEL_2 VARCHAR2(100 BYTE), 
	LEVEL_3 VARCHAR2(100 BYTE), 
	LEVEL_4 VARCHAR2(100 BYTE), 
	LEVEL_5 VARCHAR2(100 BYTE), 
	CTRL_ATYPE VARCHAR2(1 BYTE), 
	IE_BS_TYPE VARCHAR2(1 BYTE), 
	TRNS_IE_BAL_YN VARCHAR2(1 BYTE) DEFAULT 'N', 
	SCHED_NO NUMBER(5,0), 
	GPOST_Y_N VARCHAR2(1 BYTE), 
	SECURITY_Y_N VARCHAR2(1 BYTE), 
	PERMITTED_Y_N VARCHAR2(1 BYTE) DEFAULT 'N', 
	TAXABLE_Y_N VARCHAR2(1 BYTE) DEFAULT 'N', 
	CCYREVAL_Y_N VARCHAR2(1 BYTE) DEFAULT 'N', 
	INTERFACE_AC VARCHAR2(100 BYTE), 
	STATUS VARCHAR2(4 BYTE), 
	INP_DT_TM VARCHAR2(16 BYTE)
   ) SEGMENT CREATION IMMEDIATE 
  PCTFREE 10 PCTUSED 40 INITRANS 1 MAXTRANS 255 NOCOMPRESS LOGGING
  STORAGE(INITIAL 65536 NEXT 65536 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1 BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE HHI_TBS ;
--------------------------------------------------------
--  DDL for Index CT_INDX_ACLEVEL
--------------------------------------------------------

  CREATE INDEX CT_INDX_ACLEVEL ON HI_ACLEVEL_PREV (AMC_CODE, SCHEME, LEVEL_1, LEVEL_2, LEVELS, IE_BS_TYPE, RECTYPE, CURRNO) 
  PCTFREE 10 INITRANS 2 MAXTRANS 255 COMPUTE STATISTICS 
  STORAGE(INITIAL 65536 NEXT 65536 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1 BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE HHINDX_TBS ;
--------------------------------------------------------
--  DDL for Index CT_PK_ACLEVEL
--------------------------------------------------------

  CREATE UNIQUE INDEX CT_PK_ACLEVEL ON HI_ACLEVEL_PREV (AMC_CODE, SCHEME, CONTROL_AC, RECTYPE, CURRNO) 
  PCTFREE 10 INITRANS 2 MAXTRANS 255 COMPUTE STATISTICS 
  STORAGE(INITIAL 65536 NEXT 65536 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1 BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE HHINDX_TBS ;


