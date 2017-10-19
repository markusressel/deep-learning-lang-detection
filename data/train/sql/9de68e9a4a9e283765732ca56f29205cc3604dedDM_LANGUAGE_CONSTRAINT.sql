--------------------------------------------------------
--  Constraints for Table DM_LANGUAGE
--------------------------------------------------------

  ALTER TABLE "DM_LANGUAGE" ADD CONSTRAINT "CKC_RL_DM_LANGU" CHECK (RL IS NULL OR (RL IN ('R','L'))) ENABLE;
 
  ALTER TABLE "DM_LANGUAGE" ADD CONSTRAINT "PK_DM_LANGUAGE" PRIMARY KEY ("LANGUAGE_ID")
  USING INDEX PCTFREE 10 INITRANS 2 MAXTRANS 255 COMPUTE STATISTICS 
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1 BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "USERS"  ENABLE;
 
  ALTER TABLE "DM_LANGUAGE" MODIFY ("STATUS_ID" NOT NULL ENABLE);
 
  ALTER TABLE "DM_LANGUAGE" MODIFY ("LANGUAGE_ID" NOT NULL ENABLE);