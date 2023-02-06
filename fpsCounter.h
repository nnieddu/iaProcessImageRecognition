class FpsCounter
{
protected:
	unsigned int fps;
	clock_t current_ticks;
	clock_t delta_ticks;

public:
	FpsCounter() : fps(0), current_ticks(0), delta_ticks(0) {}

void update()
	{
    delta_ticks = clock() - current_ticks;
    if (delta_ticks > 0)
        fps = CLOCKS_PER_SEC / delta_ticks;
	}

	unsigned int updateAndGet()
	{
    delta_ticks = clock() - current_ticks;
    if (delta_ticks > 0)
        fps = CLOCKS_PER_SEC / delta_ticks;
		return fps;
	}

	void setCurrentTick() { current_ticks = clock(); }

	unsigned int get() const { return fps; }
};