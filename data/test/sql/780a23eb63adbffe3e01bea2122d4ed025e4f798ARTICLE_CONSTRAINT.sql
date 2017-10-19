--------------------------------------------------------
--  Constraints for Table ARTICLE
--------------------------------------------------------

  ALTER TABLE "BROCARCL"."ARTICLE" ADD CONSTRAINT "PK_ARTICLE" PRIMARY KEY ("IDARTICLE")
  USING INDEX PCTFREE 10 INITRANS 2 MAXTRANS 255 COMPUTE STATISTICS 
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1 BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "USERS"  ENABLE;
 
  ALTER TABLE "BROCARCL"."ARTICLE" MODIFY ("IDARTICLE" NOT NULL ENABLE);
 
  ALTER TABLE "BROCARCL"."ARTICLE" MODIFY ("IDCOMMANDE" NOT NULL ENABLE);
 
  ALTER TABLE "BROCARCL"."ARTICLE" MODIFY ("IDALBUM" NOT NULL ENABLE);