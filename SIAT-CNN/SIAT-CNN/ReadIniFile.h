#pragma once

#include "Preferences.h"
#include "StdAfx.h"

class CReadIniFile : public CWinApp
{
public:
	CReadIniFile(void);
	~CReadIniFile(void);
	virtual BOOL InitInstance(void);

public:
	//CPreferences m_Preferences;   //全局参数类，用于各模块（如功能页面）的进程管理与模块间数据共享
	CString m_sModulePath;        //保存进程运行的可执行程序的当前路径
};

