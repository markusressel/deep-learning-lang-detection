--------------------------------------------------------
--  DDL for Table TERMINER_ETAPE
--------------------------------------------------------

  CREATE TABLE "G11_FLIGHT"."TERMINER_ETAPE" 
   (	"TOUR_ANNEE" NUMBER(4,0), 
	"PART_NUM" NUMBER(3,0), 
	"ETAPE_NUM" NUMBER(2,0), 
	"CYCLISTE_NOM" VARCHAR2(40 BYTE), 
	"CYCLISTE_PRENOM" VARCHAR2(30 BYTE), 
	"CYCLISTE_PAYS" VARCHAR2(50 BYTE), 
	"CYCLISTE_DATEN" DATE, 
	"EQUIPE_NOM" VARCHAR2(50 BYTE), 
	"ETAPE_TPS" NUMBER(10,0), 
	"ETAPE_CLASS" NUMBER(3,0), 
	"ETAPE_PTS_MONT" NUMBER(3,0), 
	"ETAPE_PTS_SPRINT" NUMBER(3,0), 
	"GENE_TPS" NUMBER(10,0), 
	"GENE_CLASS" NUMBER(3,0), 
	"GENE_PTS_MONT" NUMBER(3,0), 
	"GENE_CLASS_MONT" NUMBER(3,0), 
	"GENE_PTS_SPRINT" NUMBER(3,0), 
	"GENE_CLASS_SPRINT" NUMBER(3,0), 
	"GENE_TPS_ECART" NUMBER(10,0), 
	"ETAPE_TPS_ECART" NUMBER(10,0), 
	"PT_PASS_NUM" NUMBER(3,0), 
	 CONSTRAINT "PK_TERMINER_ETAPE" PRIMARY KEY ("TOUR_ANNEE", "PART_NUM", "ETAPE_NUM") ENABLE
   ) ORGANIZATION INDEX NOCOMPRESS PCTFREE 10 INITRANS 2 MAXTRANS 255 
  TABLESPACE "BD50_DATA" 
  STORAGE(
  BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
 PCTTHRESHOLD 50
  PARTITION BY RANGE ("TOUR_ANNEE") 
 (PARTITION "TERMINER_ETAPE_2012"  VALUES LESS THAN (2012) 
  PCTFREE 10 INITRANS 2 MAXTRANS 255 
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1 BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "BD50_DATA" , 
 PARTITION "TERMINER_ETAPE_2013"  VALUES LESS THAN (2013) 
  PCTFREE 10 INITRANS 2 MAXTRANS 255 
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1 BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "BD50_DATA" , 
 PARTITION "TERMINER_ETAPE_2014"  VALUES LESS THAN (2014) 
  PCTFREE 10 INITRANS 2 MAXTRANS 255 
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1 BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "BD50_DATA" , 
 PARTITION "TERMINER_ETAPE_2015"  VALUES LESS THAN (2015) 
  PCTFREE 10 INITRANS 2 MAXTRANS 255 
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1 BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "BD50_DATA" , 
 PARTITION "TERMINER_ETAPE_2016"  VALUES LESS THAN (2016) 
  PCTFREE 10 INITRANS 2 MAXTRANS 255 
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1 BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "BD50_DATA" , 
 PARTITION "TERMINER_ETAPE_2017"  VALUES LESS THAN (2017) 
  PCTFREE 10 INITRANS 2 MAXTRANS 255 
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1 BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "BD50_DATA" , 
 PARTITION "TERMINER_ETAPE_2018"  VALUES LESS THAN (2018) 
  PCTFREE 10 INITRANS 2 MAXTRANS 255 
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1 BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "BD50_DATA" , 
 PARTITION "TERMINER_ETAPE_2019"  VALUES LESS THAN (2019) 
  PCTFREE 10 INITRANS 2 MAXTRANS 255 
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1 BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "BD50_DATA" , 
 PARTITION "TERMINER_ETAPE_2020"  VALUES LESS THAN (2020) 
  PCTFREE 10 INITRANS 2 MAXTRANS 255 
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1 BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "BD50_DATA" ) ;
