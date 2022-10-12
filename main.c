void find_edges_lr(int *brightness, int threshold)
{
    int left  = (brightness[-1][-1] + brightness[-1][0] + brightness[-1][1]) / 3;
    int right = (brightness[ 1][-1] + brightness[ 1][0] + brightness[ 1][1]) / 3;
    int avg   = (left + right) / 2;
    return (abs(left - right) > (threshold * avg))
}
