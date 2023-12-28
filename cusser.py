import llm

work_file = "example.txt"
model_name = "zephyr-7b"
suffix = ''

if __name__ == '__main__':
    print('initialing model...')
    llm.bot_model = llm.model_dict[model_name]()
    print(' init success ', llm.bot_model)

    exec_flag = 1
    while 1:
        
        while 1:
            try:
                exec_flag = int(input('Ready to generate? 1: execute, 0: quit\t'))
            except KeyboardInterrupt:
                exec_flag = 0
                break
            except:
                exec_flag = -1
            if (exec_flag == 1 or exec_flag == 0):
                break
        
        if not exec_flag:
            break
        else:
            with open(work_file, 'r') as f:
                hist = f.read()
            old_hist = hist = hist.rstrip('\n')
            try:
                out = llm.bot_model.generate(hist)
            except SyntaxError:
                continue

            print('rewriting file...')
            with open(work_file, 'w') as f:
                f.write(old_hist + out + suffix)
            print('done!\n\n=====================\n')