/* Mutual Exclusion Protocol - Peterson's Algorithm for 2 Processes */

/* Shared variables */
bool flag[2];
byte turn;
byte critical = 0;  /* Counter for processes in critical section */

/* Process 0 */
proctype process0()
{
    do
    :: true ->
        /* Entry protocol */
        flag[0] = true;
        turn = 1;
        (flag[1] == false || turn == 0);
        
        /* Critical section */
        critical++;
        assert(critical == 1);  /* Mutual exclusion property */
        critical--;
        
        /* Exit protocol */
        flag[0] = false;
        
        /* Non-critical section - some additional states */
        skip;
        skip;
    od
}

/* Process 1 */
proctype process1()
{
    do
    :: true ->
        /* Entry protocol */
        flag[1] = true;
        turn = 0;
        (flag[0] == false || turn == 1);
        
        /* Critical section */
        critical++;
        assert(critical == 1);  /* Mutual exclusion property */
        critical--;
        
        /* Exit protocol */
        flag[1] = false;
        
        /* Non-critical section - some additional states */
        skip;
        skip;
    od
}

/* Additional process to increase state space complexity */
proctype process2()
{
    byte local = 0;
    do
    :: local < 3 ->
        local++;
        skip;
    :: local >= 3 ->
        local = 0;
    od
}

/* LTL property: Eventually process 0 can enter critical section */
ltl eventual_entry { []<>(flag[0] == true) }

/* Initialize system */
init
{
    flag[0] = false;
    flag[1] = false;
    turn = 0;
    critical = 0;
    
    atomic {
        run process0();
        run process1();
        run process2();
    }
}