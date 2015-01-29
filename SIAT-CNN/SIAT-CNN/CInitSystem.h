#pragma once
class CInitSystem
{
public:
	CInitSystem(void);

public:
	CPreferences m_Preferences;
	static  GetPreferences();

public:
	~CInitSystem(void);
};

