package scheme

func LoadAll(path string) {
	LoadAchievement(path)
	LoadAward(path)
	LoadBuilding(path)
	LoadBuildingLvUp(path)
	LoadCommon(path)
	LoadDrop(path)
	LoadGLocalization(path)
	LoadHero(path)
	LoadHeroLvUp(path)
	LoadHeroRank(path)
	LoadHeroStageUp(path)
	LoadInitialMap(path)
	LoadItem(path)
	LoadKingSkill(path)
	LoadKingSkillLvUp(path)
	LoadMission(path)
	LoadNPC(path)
	LoadObstacle(path)
	LoadRoleLvUp(path)
	LoadSkill(path)
	LoadSkillFlash(path)
	LoadSkillLvUp(path)
	LoadSkillStageUp(path)
	LoadSoldier(path)
	LoadSoldierLvUp(path)
	LoadSoldierStageUp(path)
	LoadStage(path)
	LoadStageTask(path)
	LoadSign(path)
	LoadMall(path)
	LoadPlunderAward(path)
	LoadArenaBoss(path)
	LoadArenaRank(path)
	LoadArenaShop(path)
	LoadChallengeStage(path)
	LoadChallengeTroops(path)
	LoadModuleOpen(path)
	LoadActivity(path)
	LoadActivityOrder(path)
	LoadActivityLevelBag(path)
	LoadActivityGrowFund(path)
	LoadActivityTotalPay(path)
	LoadRecharge(path)
}

func Process() {
	AwardProcess()
	BuildingLvUpProcess()
	DropProcess()
	HeroLvUpProcess()
	HeroStageUpProcess()
	HeroRankProcess()
	RoleLvUpProcess()
	SkillLvUpProcess()
	SignProcess()
	SoldierLvUpProcess()
	SoldierStageUpProcess()
	PlunderAwardProcess()
	ArenaRankProcess()
	ArenaShopProcess()
	ChallengeStageProcess()
	ChallengeTroopProcess()
	StageProcess()
}
