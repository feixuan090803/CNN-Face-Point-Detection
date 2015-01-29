#include <iostream>
#include <afx.h>

#include "CCreateNetwork.h"
#include "ReadIniFile.h"

void SetThreadName( DWORD dwThreadID, LPCSTR szThreadName);

//extern CReadIniFile theApp;

int main()
{
	// set main thread's name (useful for debugging etc)
	char str[] = "MAIN";  // must use chars, not TCHARs, for SetThreadname function
	SetThreadName( -1, str );

	// Seed the random-number generator with current time so that
    // the numbers will be different every time we run.
	srand( (unsigned)time( NULL ) );
	
	AfxGetApp()->InitInstance();
	CCreateNetwork crtNetwork;
	

	int iSelect;
	cout<<"******************************************************"<<endl;
	cout<<"本系统提供以下操作:"<<endl<<endl;
	cout<<"	1、训练网络"<<endl;
	cout<<"	2、测试网络"<<endl;
	cout<<"	0、退出"<<endl;
	cout<<"******************************************************"<<endl;

	cout<<"输入需要进行的操作(0-2):";
	
	while(cin>>iSelect)
	{
		if(iSelect==0)
		{
			return 0;
		}
		else if(iSelect==1)
		{
			////// 实现BP后向调整过程
			char ch;
			bool flag;  
			cout<<endl<<"开始训练前是否需要载入预设权重？[Y(yes) or N(no)]: ";
			while(cin>>ch)
			{
				if(ch == 'Y' || ch == 'y')
				{
					flag=true;
					break;
				}
				else if(ch == 'N' || ch == 'n')
				{
					flag=false;
					break;
				}
				else
				{
					cout<<endl<<"不合法的输入，请重新输入！！"<<endl;
					cout<<endl<<"开始训练前是否需要载入预设权重？[Y(yes) or N(no)]: ";
				}
			}

			crtNetwork.InitNetwork(flag);
			
			crtNetwork.BackPropagation();   // 初始化BP过程的相关参数

			cout<<"神经网络训练完成！"<<endl<<endl;
		}
		else if(iSelect==2)
		{
			////// 实现FP前向计算过程
			char ch;
			bool flag;  
			cout<<endl<<"开始训练前是否需要载入预设权重？[Y(yes) or N(no)]: ";
			while(cin>>ch)
			{
				if(ch == 'Y' || ch == 'y')
				{
					flag=true;
					break;
				}
				else if(ch == 'N' || ch == 'n')
				{
					flag=false;
					break;
				}
				else
				{
					cout<<endl<<"不合法的输入，请重新输入！！"<<endl;
					cout<<endl<<"开始训练前是否需要载入预设权重？[Y(yes) or N(no)]: ";
				}
			}

			crtNetwork.InitNetwork(flag);
			bool rt=crtNetwork.ForwardPropagation();
			if(rt == true)
			{
				cout<<endl<<"*************************************************"<<endl;
				cout<<"人脸特征点检测过程完成！";
				cout<<endl<<"*************************************************"<<endl<<endl;
			}
			else
			{
				cout<<"用户操作非法，当前操作中止！"<<endl<<endl;				
			}

		}
		else
		{
			cout<<"输入有误，请重新选择操作！"<<endl;
		}

		cout<<"******************************************************"<<endl;
		cout<<"本系统提供以下操作:"<<endl<<endl;
		cout<<"	1、训练过程"<<endl;
		cout<<"	2、测试过程"<<endl;
		cout<<"	0、退出"<<endl;
		cout<<"******************************************************"<<endl;

		cout<<"输入需要进行的操作(0-2):";
	}
	return 0;
}


void SetThreadName( DWORD dwThreadID, LPCSTR szThreadName)
{
	
	struct THREADNAME_INFO
	{
		DWORD dwType; // must be 0x1000
		LPCSTR szName; // pointer to name (in user addr space)
		DWORD dwThreadID; // thread ID (-1=caller thread)
		DWORD dwFlags; // reserved for future use, must be zero
	} ;
	
	THREADNAME_INFO info;
	info.dwType = 0x1000;
	info.szName = szThreadName;
	info.dwThreadID = dwThreadID;
	info.dwFlags = 0;
	
	__try
	{
		RaiseException( 0x406D1388, 0, sizeof(info)/sizeof(DWORD), (DWORD*)&info );
	}
	__except(EXCEPTION_CONTINUE_EXECUTION)
	{
	}
}
